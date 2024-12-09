#!/usr/bin/python

from pathlib import Path
from models.Stack import Stack
import utils.network_utils
from core.train import Trainer
from losses.multi_loss import *
from utils import log
import glob
from tensorboardX import SummaryWriter

class Tester(Trainer):
    def __init__(self, opt, output_dir):
        # Initial settings
        self.opt = opt
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.deblurnet = Stack(device = self.device, **opt.network)

        # Initialize Tensorboard
        if opt.eval.use_tensorboard:
            self.tb_writer = SummaryWriter(output_dir)
        else:
            self.tb_writer = None

        # Initialize Evaluator
        from core.evaluation import Evaluation
        self.evaluation = Evaluation(self.opt, self.output_dir, self.tb_writer)

        # self.deblurnet = torch.nn.DataParallel(self.deblurnet).to(self.device)
        self.deblurnet = self.deblurnet.to(self.device)



        # Read Weights Path List
        if '*' in self.opt.test_weights:
            self.weight_path_l = sorted(glob.glob(self.opt.test_weights))
            prefix, suffix = opt.test_weights.split('*')
            self.epoch_l = [weight_path.replace(prefix, '').replace(suffix, '')  for weight_path in self.weight_path_l]
        else:
            self.weight_path_l = [self.opt.test_weights]
            self.epoch_l = [str(self.init_epoch).zfill(4)]


    def _load_test_weights(self, weight_path):

        log.info(f'{dt.now()} Recovering from {weight_path} ...')
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
        self.init_epoch = checkpoint['epoch_idx']   
    
    def test(self):

        for epoch, weight_path in zip(self.epoch_l, self.weight_path_l):
            self._load_test_weights(weight_path)
            self.evaluation.eval_all_dataset(self.deblurnet, self.output_dir, int(epoch))