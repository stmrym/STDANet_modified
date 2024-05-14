#!/usr/bin/python


import os
import glob
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.packing
from models.Stack import Stack
# from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from core.train import train
from core.evaluation import evaluation
import logging
from losses.multi_loss import *
from utils import log

def get_weights(path, multi_file = True):
    if multi_file:
        weights = sorted(glob.glob(os.path.join(path, 'ckpt-epoch-*.pth.tar')))
    else:
        weights = [path]
    return weights



def  bulid_net(cfg,output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    if cfg.NETWORK.PHASE in ['train', 'resume']:

        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
            utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
            utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
            utils.data_transforms.RandomVerticalFlip(),
            utils.data_transforms.RandomHorizontalFlip(),
            # utils.data_transforms.RandomColorChannel(),
            utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
            utils.data_transforms.ToTensor(),
        ])

        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
            utils.data_transforms.ToTensor(),
        ])

    elif cfg.NETWORK.PHASE in ['test']:
        test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.ToTensor(),
        ])
    
    # Set up networks
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    deblurnet = Stack(  
                        network_arch = cfg.NETWORK.DEBLURNETARCH, 
                        use_stack = cfg.NETWORK.USE_STACK, 
                        n_sequence = cfg.NETWORK.INPUT_LENGTH, 
                        in_channels = cfg.NETWORK.INPUT_CHANNEL,
                        n_feat = cfg.NETWORK.NUM_FEAT,
                        out_channels = cfg.NETWORK.OUTPUT_CHANNEL,
                        n_resblock = cfg.NETWORK.NUM_RESBLOCK,
                        kernel_size = cfg.NETWORK.KERNEL_SIZE,
                        sobel_out_channels = cfg.NETWORK.SOBEL_OUT_CHANNEL,
                        device = device
                        )
    
    log.info(f'{dt.now()} Parameters in {cfg.NETWORK.DEBLURNETARCH}: {utils.network_utils.count_parameters(deblurnet)}.')
    log.info(f'Loss: {cfg.LOSS_DICT_LIST} ')

    # Initialize weights of networks
    # Train or resume phase
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # Set up solver
        base_params = []
        motion_branch_params = []
        attention_params = []

        for name,param in deblurnet.named_parameters():
            if 'reference_points' in name or 'sampling_offsets' in name:
                if param.requires_grad:


                    attention_params.append(param)
            # elif "spynet" in name or "flow_pwc" in name or "flow_net" in name:
            elif 'motion_branch' in name or 'motion_out' in name:
                if param.requires_grad:
                    # Fix weigths for motion estimator
                    if not cfg.LOSS.MOTION_REQUIRES_GRAD:
                        log.info(f'Motion requires grad ... False')
                        param.requires_grad = False
                    motion_branch_params.append(param)
            elif 'edge_extractor' in name and not cfg.LOSS.SOBEL_REQUIRES_GRAD:
                param.requires_grad = False

            else:
                if param.requires_grad:
                    base_params.append(param)
        
        optim_param = [
                {'params':base_params,'initial_lr':cfg.TRAIN.LEARNING_RATE,"lr":cfg.TRAIN.LEARNING_RATE},
                {'params':motion_branch_params,'initial_lr':cfg.TRAIN.LEARNING_RATE,"lr":cfg.TRAIN.LEARNING_RATE},
                {'params':attention_params,'initial_lr':cfg.TRAIN.LEARNING_RATE*0.01,"lr":cfg.TRAIN.LEARNING_RATE*0.01},
            ]
        deblurnet_solver = torch.optim.Adam(optim_param,lr=cfg.TRAIN.LEARNING_RATE,
                                            betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

        # Load pretrained model if exists
        init_epoch       = 0
        Best_Epoch       = -1
        Best_Img_PSNR    = 0
        
        if cfg.CONST.WEIGHTS != '':
            log.info(f'{dt.now()} Recovering from {cfg.CONST.WEIGHTS} ...')
            
            checkpoint = torch.load(os.path.join(cfg.CONST.WEIGHTS),map_location='cpu')

            deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
            deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
            for state in deblurnet_solver.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            # deblurnet_lr_scheduler.load_state_dict(checkpoint['deblurnet_lr_scheduler'])
            
            init_epoch = checkpoint['epoch_idx'] + 1
            Best_Img_PSNR = checkpoint['Best_Img_PSNR']
            Best_Epoch = checkpoint['Best_Epoch']

        
        deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                    milestones=cfg.TRAIN.LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.LR_DECAY,last_epoch=(init_epoch))
        
        if torch.cuda.is_available():
            deblurnet = torch.nn.DataParallel(deblurnet).cuda()


        ckpt_dir      = os.path.join(output_dir, 'checkpoints')
        visualize_dir = os.path.join(output_dir, 'visualization')

        # Training

        train_writer = SummaryWriter(output_dir)

        train(cfg = cfg, 
            init_epoch = init_epoch,
            train_transforms = train_transforms, val_transforms = val_transforms,
            deblurnet = deblurnet, deblurnet_solver = deblurnet_solver, 
            deblurnet_lr_scheduler = deblurnet_lr_scheduler,
            ckpt_dir = ckpt_dir, visualize_dir = visualize_dir,
            tb_writer = train_writer, Best_Img_PSNR = Best_Img_PSNR, Best_Epoch = Best_Epoch)  
    
    # Test phase
    elif cfg.NETWORK.PHASE in ['test']:
        if torch.cuda.is_available():
            deblurnet = torch.nn.DataParallel(deblurnet).cuda()
        
            # Load pretrained model if exists
        weights = get_weights(cfg.CONST.WEIGHTS, multi_file = False)
        test_writer = SummaryWriter(output_dir) if cfg.EVAL.USE_TENSORBOARD else None

        for weight in weights:

            epoch = weight.split('ckpt-epoch-')[-1].split('.pth')[0]
            log.info(f'{dt.now()} Recovering from {weight} ...')     
            checkpoint = torch.load(os.path.join(weight),map_location='cpu')
            if isinstance(deblurnet, torch.nn.DataParallel):
                deblurnet.module.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})    
            else:
                deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
            
            # Test for each dataset list
            for test_dataset_name, test_image_blur_path, test_image_clear_path, test_json_file_path\
                in zip(cfg.DATASET.TEST_DATASET_LIST, cfg.DIR.TEST_IMAGE_BLUR_PATH_LIST, cfg.DIR.TEST_IMAGE_CLEAR_PATH_LIST, cfg.DIR.TEST_JSON_FILE_PATH_LIST):
                test_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(
                    image_blur_path = test_image_blur_path, 
                    image_clear_path = test_image_clear_path,
                    json_file_path = test_json_file_path,
                    input_length = cfg.NETWORK.INPUT_LENGTH)
                
                if len(weights) != 1:
                    save_dir = os.path.join(output_dir, test_dataset_name, epoch)
                else:
                    save_dir = os.path.join(output_dir, test_dataset_name)

                _, _ = evaluation(cfg = cfg, 
                    eval_dataset_name = test_dataset_name,
                    save_dir = save_dir,
                    eval_loader = test_loader,
                    eval_transforms = test_transforms,
                    deblurnet = deblurnet,
                    epoch_idx = int(epoch),
                    tb_writer = test_writer)      