#!/usr/bin/python

from utils import log
import matplotlib
# from models import deformable_transformer
import os
import re

import torch
matplotlib.use('Agg')

from datetime import datetime as dt
from config import cfg
from core.build import bulid_net
import warnings
warnings.filterwarnings("ignore") 
# torch.manual_seed(1)

def main():
    
    if cfg.NETWORK.PHASE == 'resume':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'train',cfg.CONST.WEIGHTS.split("/")[-3])
        print_log  = os.path.join(output_dir, 'print.log')

    elif  cfg.NETWORK.PHASE == 'test':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'test', cfg.CONST.DEBUG_PREFIX + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG + '_' + re.split('[/.]', cfg.CONST.WEIGHTS)[-3])
        print_log    = os.path.join(output_dir, 'print.log')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    elif cfg.NETWORK.PHASE == 'train':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'train', cfg.CONST.DEBUG_PREFIX + timestr + '_' + cfg.NETWORK.DEBLURNETARCH + "_" + cfg.NETWORK.TAG) # changed to use timestr
        log_dir      = os.path.join(output_dir, 'logs')
        ckpt_dir     = os.path.join(output_dir, 'checkpoints')
        print_log    = os.path.join(output_dir, 'print.log')
        data_log_dirs = [log_dir,ckpt_dir]
        for dir_exit in data_log_dirs:
            if not os.path.exists(dir_exit):
                os.makedirs(dir_exit)
    else:
        print('Invalid NETWORK.PHASE!')
        exit()

    log.setFileHandler(print_log,mode="w")
    # Print config
    log.info('Use config:')
    log.info(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    log.info(f' Output_dir： {output_dir}')

    # Setup Network & Start train/test process
    bulid_net(
        cfg = cfg,
        output_dir = output_dir)

if __name__ == '__main__':

    main()
