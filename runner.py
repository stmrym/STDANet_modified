#!/usr/bin/python

from utils import log
import matplotlib
matplotlib.use('Agg')
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
""" from visualizer import get_local
get_local.activate() """
import numpy as np
import random
from pathlib import Path
import argparse
import shutil
import yaml
from box import Box
from datetime import datetime as dt
# import warnings
# warnings.filterwarnings("ignore") 

def main():

    # Loading opt from .yml file
    parser = argparse.ArgumentParser(description='STDAN modified')
    parser.add_argument('config', help='Please set .yml file')
    args = parser.parse_args()
    with open(args.config, mode='r') as f:
        opt = yaml.safe_load(f)
    opt = Box(**opt)

    if opt.phase == 'resume':
        # output_dir = Path(opt.exp_path) / 'train' / (opt.weights).split('/')[-3]
        output_dir = Path(opt.weights.split('checkpoints')[0])

    elif opt.phase == 'test':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')
        output_dir = Path(opt.exp_path) / 'test' / Path(timestr + '_' + opt.prefix)
        os.makedirs(output_dir, exist_ok=True)        

    elif opt.phase == 'train':
        timestr = dt.now().isoformat(timespec='seconds').replace(':', '')        
        network_name = opt.network.arch + '_Stack' if opt.network.use_stack else opt.network.arch
        exp_name = timestr + '_' + network_name + '_' + '_'.join(opt.dataset.train.keys())
        output_dir = Path(opt.exp_path) / 'train' / exp_name # changed to use timestr    
        ckpt_dir = output_dir / 'checkpoints'

        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copy(args.config, output_dir / f'{exp_name}.yaml')
        
    else:
        print('Invalid NETWORK.PHASE!')
        exit()

    print_log = output_dir / 'print.log'
    log.setFileHandler(print_log,mode='a')

    # Set GPU to use
    if type(opt.device) == str and not opt.device == 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    
    import torch
    from core.train import Trainer
    from core.test import Tester
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    # torch.autograd.set_detect_anomaly(True)   # detect backward anomaly, only use in debug
    
    log.info('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))
    log.info(f' Output_dir: {output_dir}')

    # Setup Train
    if opt.phase in ['train', 'resume']:
        trainer = Trainer(opt, output_dir)
        trainer.train()  
    
    # Setup Test
    elif opt.phase in ['test']:
        tester = Tester(opt, output_dir)
        tester.test()


if __name__ == '__main__':
    main()
