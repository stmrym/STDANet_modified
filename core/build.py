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



def  bulid_net(opt, output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    if opt.phase in ['train', 'resume']:
        
        train_transform_l = []
        for transform_name, transform_args in opt.train_transform.items():
            transform = getattr(utils.data_transforms, transform_name)
            train_transform_l.append(transform(**transform_args))
        train_transform_l.extend(utils.data_transforms.ToTensor())

        train_transforms = utils.data_transforms.Compose(train_transform_l)
        # train_transforms = utils.data_transforms.Compose([
        #     utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        #     utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        #     utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        #     utils.data_transforms.RandomVerticalFlip(),
        #     utils.data_transforms.RandomHorizontalFlip(),
        #     # utils.data_transforms.RandomColorChannel(),
        #     utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        #     ,
        # ])

    eval_transform_l = []
    for transform_name, transform_args in opt.eval_transform.items():
        transform = getattr(utils.data_transforms, transform_name)
        eval_transform_l.append(transform(**transform_args))
    eval_transform_l.extend(utils.data_transforms.ToTensor())
    eval_transforms = utils.data_transforms.Compose(eval_transform_l)
  
    # Set up networks
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    deblurnet = Stack(device = device, **opt.network)
    
    log.info(f'{dt.now()} Parameters in {opt.network.arch}: {utils.network_utils.count_parameters(deblurnet)}.')
    log.info(f'Loss: {opt.loss.keys()} ')

    # Initialize weights of networks
    # Train or resume phase
    if opt.phase in ['train', 'resume']:
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
                    if not opt.train.motion_requires_grad:
                        log.info(f'Motion requires grad ... False')
                        param.requires_grad = False
                    motion_branch_params.append(param)
            elif 'edge_extractor' in name and not opt.train.sobel_requires_grad:
                param.requires_grad = False

            else:
                if param.requires_grad:
                    base_params.append(param)
        
        lr = opt.train.optimization.learning_rate
        optim_param = [
                {'params':base_params,'initial_lr':lr,"lr":lr},
                {'params':motion_branch_params,'initial_lr':lr,"lr":lr},
                {'params':attention_params,'initial_lr':lr*0.01,"lr":lr*0.01},
            ]
        deblurnet_solver = torch.optim.Adam(optim_param,lr = lr,
                                            betas=(opt.train.optimization.momentum, opt.train.optimization.beta))

        # Load pretrained model if exists
        init_epoch       = 0
        Best_Epoch       = -1
        Best_Img_PSNR    = 0
        
        if opt.weights != '':
            log.info(f'{dt.now()} Recovering from {opt.weights} ...')
            
            checkpoint = torch.load(os.path.join(opt.weights),map_location='cpu')

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
                                                                    milestones=opt.train.optimization.lr_milestones,
                                                                    gamma=opt.train.optimization.lr_decay,last_epoch=(init_epoch))
        
        if torch.cuda.is_available():
            deblurnet = torch.nn.DataParallel(deblurnet).cuda()

        ckpt_dir      = os.path.join(output_dir, 'checkpoints')
        visualize_dir = os.path.join(output_dir, 'visualization')

        # Training

        train_writer = SummaryWriter(output_dir)

        train(opt = opt, 
            init_epoch = init_epoch,
            train_transforms = train_transforms, val_transforms = eval_transforms,
            deblurnet = deblurnet, deblurnet_solver = deblurnet_solver, 
            deblurnet_lr_scheduler = deblurnet_lr_scheduler,
            ckpt_dir = ckpt_dir, visualize_dir = visualize_dir,
            tb_writer = train_writer, Best_Img_PSNR = Best_Img_PSNR, Best_Epoch = Best_Epoch)  
    
    # Test phase
    elif opt.phase in ['test']:
        if torch.cuda.is_available():
            deblurnet = torch.nn.DataParallel(deblurnet).cuda()
        
            # Load pretrained model if exists
        weights = get_weights(opt.weights, multi_file = False)
        test_writer = SummaryWriter(output_dir) if opt.eval.use_tensorboard else None

        for weight in weights:

            epoch = weight.split('ckpt-epoch-')[-1].split('.pth')[0]
            log.info(f'{dt.now()} Recovering from {weight} ...')     
            checkpoint = torch.load(os.path.join(weight),map_location='cpu')
            if isinstance(deblurnet, torch.nn.DataParallel):
                deblurnet.module.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})    
            else:
                deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
            
            # Test for each dataset list
            test_dataset = opt.dataset.test
            for test_dataset_name, test_image_blur_path, test_image_clear_path, test_json_file_path\
                in zip(test_dataset.dataset, test_dataset.blur_path, test_dataset.sharp_path, test_dataset.json_path):
                test_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(
                    image_blur_path = test_image_blur_path, 
                    image_clear_path = test_image_clear_path,
                    json_file_path = test_json_file_path,
                    input_length = opt.network.input_length)
                
                if len(weights) != 1:
                    save_dir = os.path.join(output_dir, test_dataset_name, epoch)
                else:
                    save_dir = os.path.join(output_dir, test_dataset_name)

                _, _ = evaluation(opt = opt, 
                    eval_dataset_name = test_dataset_name,
                    save_dir = save_dir,
                    eval_loader = test_loader,
                    eval_transforms = eval_transforms,
                    deblurnet = deblurnet,
                    epoch_idx = int(epoch),
                    tb_writer = test_writer)      
                