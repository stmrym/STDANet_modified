#!/usr/bin/python

from utils import log
import matplotlib
""" from visualizer import get_local
get_local.activate() """
# from models import deformable_transformer
import os
import re

import torch
matplotlib.use('Agg')

from datetime import datetime as dt
from config import cfg
import warnings
warnings.filterwarnings("ignore") 
# torch.manual_seed(1)

def main():
    

    # if args.data_path is not None:
    #     cfg.DIR.DATASET_ROOT = args.data_path
    #     if cfg.DATASET.DATASET_NAME == 'DVD':
    #         cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/VideoDeblur.json'
    #         cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/input/%s.jpg')
    #         cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/GT/%s.jpg')
    #     if cfg.DATASET.DATASET_NAME == 'GOPRO':
    #         cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/GoproDeblur.json'
    #         cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/blur_gamma/%s.png')
    #         cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/sharp/%s.png')
    #     if cfg.DATASET.DATASET_NAME in ['BSD_1ms8ms','BSD_2ms16ms','BSD_3ms24ms']:

    #         cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/Blur/RGB/%s.png')
    #         cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/Sharp/RGB/%s.png')
    #         if cfg.DATASET.DATASET_NAME == 'BSD_1ms8ms':
    #             cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_1ms8msDeblur.json'
    #         elif cfg.DATASET.DATASET_NAME == 'BSD_2ms16ms':
    #             cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_2ms16msDeblur.json'
    #         elif cfg.DATASET.DATASET_NAME == 'BSD_3ms24ms':
    #             cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/BSD_3ms24msDeblur.json'
    #     if cfg.DATASET.DATASET_NAME == 'REDS_RR':
    #         # cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/REDS_RR.json'
    #         # cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/REDS_RR_train_val_test.json'
    #         # cfg.DIR.DATASET_JSON_FILE_PATH = './datasets/REDS_RR_val.json'
    #         cfg.DIR.DATASET_JSON_FILE_PATH = args.json_path
    #         cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/%s/input/%s.png')
    #         cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/%s/GT/%s.png')   
    #     if cfg.DATASET.DATASET_NAME == 'original':
    #         cfg.DIR.DATASET_ROOT = args.data_path
    #         cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/input/%s/%s.png')
    #         cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/GT/%s/%s.png')
    
    # #####################################################################################
    
    # #####################################################################################

    
    if cfg.NETWORK.PHASE == 'resume':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"train",cfg.CONST.WEIGHTS.split("/")[-3], '%s')
        print_log    = output_dir % 'print.log'

    elif  cfg.NETWORK.PHASE == "test":
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"test", cfg.CONST.DEBUG_PREFIX + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG + '_' + re.split('[/.]', cfg.CONST.WEIGHTS)[-3], '%s')
        print_log    = output_dir % 'print.log'
        dir_exit = os.path.join(cfg.DIR.OUT_PATH,"test", cfg.CONST.DEBUG_PREFIX + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG + '_' + re.split('[/.]', cfg.CONST.WEIGHTS)[-3])
        if not os.path.exists(dir_exit):
            os.makedirs(dir_exit)
    else:
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        output_dir = os.path.join(cfg.DIR.OUT_PATH,"train", cfg.CONST.DEBUG_PREFIX + timestr + '_' + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG, '%s') # changed to use timestr
        log_dir      = output_dir % 'logs'
        ckpt_dir     = output_dir % 'checkpoints'
        print_log    = output_dir % 'print.log'
        data_log_dirs = [log_dir,ckpt_dir]
        for dir_exit in data_log_dirs:
            if not os.path.exists(dir_exit):
                os.makedirs(dir_exit)
    
    
    log.setFileHandler(print_log,mode="w")
    # Print config
    log.info('Use config:')
    log.info(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    from core.build import bulid_net
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    
    # Setup Network & Start train/test process
    bulid_net(cfg,output_dir)

if __name__ == '__main__':

    main()
