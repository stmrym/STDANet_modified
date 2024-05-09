#!/usr/bin/python


import os
import torch.backends.cudnn
import torch.utils.data

from models.Stack import Stack
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.packing
import glob
import importlib
# from models.STDAN_Stack import STDAN_Stack
# from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from datetime import datetime as dt
from tensorboardX import SummaryWriter
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

def test(cfg,output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    utils.data_transforms.ToTensor(),
    ])
    
    # Set up networks
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    # deblurnet = module.__dict__[cfg.NETWORK.DEBLURNETARCH](cfg = cfg)
    deblurnet = Stack(  
                    network_arch=cfg.NETWORK.DEBLURNETARCH, 
                    use_stack=cfg.NETWORK.USE_STACK, 
                    n_sequence=cfg.DATA.INPUT_LENGTH, 
                    device = device)
    
    if torch.cuda.is_available():
        deblurnet = torch.nn.DataParallel(deblurnet).cuda()
    
    log.info(f'{dt.now()} Parameters in {cfg.NETWORK.DEBLURNETARCH}: {utils.network_utils.count_parameters(deblurnet)}.')
    log.info(f'Loss: {cfg.LOSS_DICT_LIST} ')

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
                input_length = cfg.DATA.INPUT_LENGTH)
            
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