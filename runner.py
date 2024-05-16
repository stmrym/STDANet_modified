#!/usr/bin/python

from numpy.lib.utils import info
from utils import log

import matplotlib
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
""" from visualizer import get_local
get_local.activate() """
import numpy as np
import re
import random
import importlib
import argparse
import shutil
matplotlib.use('Agg')
from datetime import datetime as dt
# import warnings
# warnings.filterwarnings("ignore") 

def main():

    parser = argparse.ArgumentParser(description='STDAN modified')
    parser.add_argument('config', help='Please set [PATH_TO_CONFIG].py name (e.g., "config/config_1")')

    # Loading [CONFIG].py
    args = parser.parse_args()
    config_import_name = args.config.replace('/', '.')

    config = importlib.import_module(config_import_name)
    cfg = config.cfg

    if cfg.NETWORK.PHASE == 'resume':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'train',cfg.CONST.WEIGHTS.split('/')[-3])
        print_log  = os.path.join(output_dir, 'print.log')

    elif  cfg.NETWORK.PHASE == 'test':
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'test', cfg.CONST.PREFIX)
        print_log    = os.path.join(output_dir, 'print.log')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        

    elif cfg.NETWORK.PHASE == 'train':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'train', cfg.CONST.PREFIX + timestr + '_' + cfg.NETWORK.DEBLURNETARCH + '_' + '_'.join(cfg.DATASET.TRAIN_DATASET_LIST)) # changed to use timestr
        
        ckpt_dir     = os.path.join(output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        print_log    = os.path.join(output_dir, 'print.log')
        shutil.copy(args.config + '.py', os.path.join(output_dir, timestr + '_config.py'))
        
    else:
        print('Invalid NETWORK.PHASE!')
        exit()

    log.setFileHandler(print_log,mode='a')

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CONST.DEVICE
    
    
    import torch
    from core.build import bulid_net
    
    random.seed(cfg.CONST.SEED)
    np.random.seed(cfg.CONST.SEED)
    torch.manual_seed(cfg.CONST.SEED)
    torch.cuda.manual_seed(cfg.CONST.SEED)
    torch.cuda.manual_seed_all(cfg.CONST.SEED)
    # torch.autograd.set_detect_anomaly(True)   # detect backward anomaly, only use in debug
    
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    log.info(f' Output_dir： {output_dir}')


    # Setup Network & Start train/test process
    bulid_net(cfg = cfg, output_dir = output_dir)


if __name__ == '__main__':

    main()
