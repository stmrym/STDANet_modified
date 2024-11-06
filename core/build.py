#!/usr/bin/python

import os
import glob
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.packing
# from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from datetime import datetime as dt
from core.train import Trainer
from losses.multi_loss import *
from utils import log

def get_weights(path, multi_file = True):
    if multi_file:
        weights = sorted(glob.glob(os.path.join(path, 'ckpt-epoch-*.pth.tar')))
    else:
        weights = [path]
    return weights


def build_transform(transform_opt):
    transform_l = []
    for name, args in transform_opt.items():
        transform = getattr(utils.data_transforms, name)
        transform_l.append(transform(**args) if args is not None else transform())
    transform_l.append(utils.data_transforms.ToTensor())
    return utils.data_transforms.Compose(transform_l)   


def  bulid_net(opt, output_dir):

    if opt.phase in ['train', 'resume']:
        # Training
        trainer = Trainer(opt, output_dir)
        trainer.train()  
    
    # Test phase
    elif opt.phase in ['test']:
    # Set up tranform
        eval_transforms = build_transform(opt.eval_transform)

        # Set up networks
        device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        deblurnet = Stack(device = device, **opt.network)
        deblurnet = torch.nn.DataParallel(deblurnet).to(device)
        
        log.info(f'{dt.now()} Parameters in {opt.network.arch}: {utils.network_utils.count_parameters(deblurnet)}.')
        log.info(f'Loss: {opt.loss.keys()} ')
        # Load pretrained model if exists
        weights = get_weights(opt.weights, multi_file=False)

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
                    input_length = opt.network.n_sequence)
                
                if len(weights) != 1:
                    save_dir = output_dir / test_dataset_name / epoch
                else:
                    save_dir = output_dir / test_dataset_name

                _, _ = evaluation(opt = opt, 
                    eval_dataset_name = test_dataset_name,
                    save_dir = save_dir,
                    eval_loader = test_loader,
                    eval_transforms = eval_transforms,
                    deblurnet = deblurnet,
                    epoch_idx = int(epoch),
                    tb_writer=None)      
                