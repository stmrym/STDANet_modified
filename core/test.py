#!/usr/bin/python

from pathlib import Path
from models.Stack import Stack
import utils.network_utils
from core.train import Trainer
from losses.multi_loss import *
from utils import log

class Tester(Trainer):
    def __init__(self, opt, output_dir):
        # Initial settings
        self.opt = opt
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.deblurnet = Stack(device = self.device, **opt.network)
        self._load_test_weights()

        log.info(f'{dt.now()} Parameters in {opt.network.arch}: {utils.network_utils.count_parameters(self.deblurnet)}.')
        log.info(f'Loss: {opt.loss.keys()} ')

    def _load_test_weights(self):
        log.info(f'{dt.now()} Recovering from {self.opt.weights} ...')
        checkpoint = torch.load(self.opt.weights, map_location='cpu')
        self.deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
        self.init_epoch = checkpoint['epoch_idx'] + 1        
    
    def test(self):
        from core.evaluation import Evaluation
        self.evaluation = Evaluation(self.opt, self.output_dir)
        self.deblurnet = torch.nn.DataParallel(self.deblurnet).to(self.device)
        
        save_dir = self.output_dir / 'visualization' / Path('epoch-' + str(self.init_epoch).zfill(4))
        self.evaluation.eval_all_dataset(self.deblurnet, save_dir, self.init_epoch)