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
import yaml
from box import Box
matplotlib.use('Agg')
from datetime import datetime as dt
# import warnings
# warnings.filterwarnings("ignore") 

def main():

    parser = argparse.ArgumentParser(description='STDAN modified')
    parser.add_argument('config', help='Please set [PATH_TO_CONFIG].py name (e.g., "config/config_1")')

    # Loading [CONFIG].py
    args = parser.parse_args()

    with open(args.config, mode='r') as f:
        opt = yaml.safe_load(f)
    opt = Box(**opt)

    if opt.phase == 'resume':
        output_dir = os.path.join(opt.exp_path, 'train', (opt.weights).split('/')[-3])
        print_log  = os.path.join(output_dir, 'print.log')

    elif opt.phase == 'test':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        output_dir = os.path.join(opt.exp_path,'test', timestr + '_' + opt.prefix)
        print_log    = os.path.join(output_dir, 'print.log')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        

    elif opt.phase == 'train':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        if opt.network.use_stack:
            output_dir = os.path.join(opt.exp_path,'train', timestr + '_' + opt.prefix + '_' 
                                    + opt.network.arch + '_Stack_' + '_'.join(opt.dataset.train.dataset)) # changed to use timestr
        else:
            output_dir = os.path.join(opt.exp_path,'train', timestr + '_' + opt.prefix + '_' 
                                    + opt.network.arch + '_' + '_'.join(opt.dataset.train.dataset)) # changed to use timestr

        ckpt_dir = os.path.join(output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        print_log = os.path.join(output_dir, 'print.log')
        shutil.copy(args.config, os.path.join(output_dir, timestr + '_config.yaml'))
        
    else:
        print('Invalid NETWORK.PHASE!')
        exit()

    log.setFileHandler(print_log,mode='a')

    # Set GPU to use
    if type(opt.device) == str and not opt.device == 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    
    
    import torch
    from core.build import bulid_net
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    # torch.autograd.set_detect_anomaly(True)   # detect backward anomaly, only use in debug
    
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    log.info(f' Output_dir: {output_dir}')

    # Setup Network & Start train/test process
    bulid_net(opt = opt, output_dir = output_dir)


if __name__ == '__main__':

    main()
