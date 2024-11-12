import os
import torch.backends
from utils import log, util
import torch.backends.cudnn
import torch.utils.data
from tensorboardX import SummaryWriter
from pathlib import Path
from models.Stack import Stack
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *

from torch.profiler import profile, record_function, ProfilerActivity


# from models.VGG19 import VGG19
# from utils.network_utils import flow2rgb
from tqdm import tqdm

class Trainer:
    def __init__(self, opt, output_dir):
        # Initial settings
        self.opt = opt
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.deblurnet = Stack(device=self.device, **opt.network)

        # Set tranform
        self.train_transforms = self._build_transform(opt.train_transform)

        log.info(f'{dt.now()} Parameters in {opt.network.arch}: {utils.network_utils.count_parameters(self.deblurnet)}.')
        log.info(f'Loss: {opt.loss.keys()} ')

        # Initialize optimizer
        self.deblurnet_solver = self._init_optimizer()

        self.init_epoch = 0

        if opt.weights is not None:
            self._load_weights()

        self.deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.deblurnet_solver,
                                                                milestones=opt.train.optimization.lr_milestones,
                                                                gamma=opt.train.optimization.lr_decay,last_epoch=(self.init_epoch))

        self.train_data_loader = self._build_dataloader(opt.dataset.train, 'train', self.train_transforms, opt.train_batch_size)

        if opt.eval.use_tensorboard and opt.phase in ['train', 'resume']:
            self.tb_writer = SummaryWriter(output_dir)
        else:
            self.tb_writer = None

        from core.evaluation import Evaluation
        self.evaluation = Evaluation(self.opt, self.output_dir, self.tb_writer)
        
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.visualize_dir = self.output_dir / 'visualization'


    def _build_transform(self, transform_opt):
        transform_l = []
        name_l = []
        for name, args in transform_opt.items():
            transform = getattr(utils.data_transforms, name)
            transform_l.append(transform(**args) if args is not None else transform())
            name_l.append(name)
        transform_l.append(utils.data_transforms.ToTensor())
        log.info(f'Transform... {name_l}')
        return utils.data_transforms.Compose(transform_l)   

    def _init_optimizer(self):
        base_params = []
        motion_branch_params = []
        attention_params = []

        for name,param in self.deblurnet.named_parameters():
            if param.requires_grad:
                if 'reference_points' in name or 'sampling_offsets' in name:
                    attention_params.append(param)
                # elif "spynet" in name or "flow_pwc" in name or "flow_net" in name:
                elif 'motion_branch' in name or 'motion_out' in name:
                    # Fix weigths for motion estimator
                    if not self.opt.train.motion_requires_grad:
                        log.info(f'Motion requires grad ... False')
                        param.requires_grad = False
                    motion_branch_params.append(param)
                else:
                    base_params.append(param)
        
        lr_opt = self.opt.train.optimization
        lr = lr_opt.learning_rate
        optim_param = [
                {'params':base_params, 'initial_lr':lr, 'lr':lr},
                {'params':motion_branch_params, 'initial_lr':lr, 'lr':lr},
                {'params':attention_params, 'initial_lr':lr*0.01, 'lr':lr*0.01},
            ]
        deblurnet_solver = torch.optim.Adam(optim_param,lr = lr,
                                            betas=(lr_opt.momentum, lr_opt.beta))
        return deblurnet_solver

    def _load_weights(self):
        log.info(f'{dt.now()} Recovering from {self.opt.weights} ...')
        
        checkpoint = torch.load(self.opt.weights, map_location='cpu')
        self.deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})
        self.deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])

        for state in self.deblurnet_solver.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        # deblurnet_lr_scheduler.load_state_dict(checkpoint['deblurnet_lr_scheduler'])
        self.init_epoch = checkpoint['epoch_idx'] + 1

    def _build_dataloader(self, dataset_opt, phase, transforms, batch_size):
        # Setup datasets
        dataset_list = []
        for name, dataset_dict in dataset_opt.items():
            dataset_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(input_length=self.opt.network.n_sequence, **dataset_dict)
            dataset = dataset_loader.get_dataset(transforms=transforms)
            dataset_list.append(dataset)
            log.info(f'[{phase.upper()}] Dataset [{name}] loaded. {phase} case: {len(dataset)}')

        # Concat all dataset
        all_dataset = torch.utils.data.ConcatDataset(dataset_list)
        log.info(f'[{phase.upper()}] Total {phase} case: {len(all_dataset)}')

        # Creating dataloader
        num_workers = self.opt.pop('num_workers', os.cpu_count()//torch.cuda.device_count())
        data_loader = torch.utils.data.DataLoader(
            dataset=all_dataset,
            batch_size=batch_size,
            num_workers=num_workers, pin_memory=True, shuffle=True)
        return data_loader

    def _init_epoch(self):
        # self.deblurnet = torch.nn.DataParallel(self.deblurnet).to(self.device)
        self.deblurnet = self.deblurnet.to(self.device)
        torch.backends.cudnn.benchmark = self.opt.use_cudnn_benchmark

        # Batch average meterics
        self.losses_dict = self.opt.loss.copy()
        for loss_dict in self.losses_dict.values():
            loss_dict.avg_meter = utils.network_utils.AverageMeter()
        self.total_losses = utils.network_utils.AverageMeter()

    def _before_epoch(self, epoch_idx):
        # Reset AverageMeter
        for loss_dict in self.losses_dict.values():
            loss_dict.avg_meter.reset()
        self.total_losses.reset()

        # Set tqdm
        tqdm_train = tqdm(self.train_data_loader)
        tqdm_train.set_description(f'[TRAIN] [Epoch {epoch_idx}/{self.opt.train.n_epochs}]')
        return tqdm_train

    def _after_epoch(self, epoch_idx):
        # Append epoch loss to TensorBoard
        for name, losses_dict in self.losses_dict.items():
            self.tb_writer.add_scalar(f'Loss_TRAIN/{name}', losses_dict.avg_meter.avg, epoch_idx)
        
        self.tb_writer.add_scalar('Loss_TRAIN/TotalLoss', self.total_losses.avg, epoch_idx)
        self.tb_writer.add_scalar('lr/lr', self.deblurnet_lr_scheduler.get_last_lr()[0], epoch_idx)
        self.deblurnet_lr_scheduler.step()

        # log.info(f'[TRAIN][Epoch {epoch_idx}/{self.opt.train.n_epochs}] total_losses_avg: {self.total_losses.avg}')

    def _calc_update_losses(self, output_dict, gt_seq, batch_size):
        total_loss = 0
        for loss_dict in self.losses_dict.values():
            loss = eval(loss_dict.func)(output_dict, gt_seq) * loss_dict.weight   # Calculate loss
            loss_dict.avg_meter.update(loss.item(), batch_size)    # Update loss
            total_loss += loss

        self.total_losses.update(total_loss.item(), batch_size) # Update total losses
        return total_loss

    def train(self):
                
        self._init_epoch()
        n_itr = 0
        # Start epoch
        for epoch_idx in range(self.init_epoch, self.opt.train.n_epochs):
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            tqdm_train = self._before_epoch(epoch_idx)
            # switch models to training mode
            self.deblurnet.train()

            for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_train):
                # Get data from data loader
                seq_blur  = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
                seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
                input_seq = torch.cat(seq_blur,1)
                gt_seq = torch.cat(seq_clear,1)
                

                self.deblurnet_solver.zero_grad()
                # self.deblurnet_solver.zero_grad(set_to_none=True)

                output_dict = self.deblurnet(input_seq) 
                # {'out':           {'recons_1', 'recons_2', 'recons_3', 'final'},
                #  'flow_forwards': {'recons_1', 'recons_2', 'recons_3', 'final'},
                #  'flow_backwards':{'recons_1', 'recons_2', 'recons_3', 'final'},
                #  ...}
                
                # Calculate & update loss
                total_loss = self._calc_update_losses(output_dict, gt_seq, self.opt.train_batch_size)
                
                total_loss.backward()
                self.deblurnet_solver.step()

                n_itr += 1
                # Tick / tock
                tqdm_train.set_postfix_str(f'total_loss {total_loss:.3f}, total_losses_avg {self.total_losses.avg:.3f}')
                
            self._after_epoch(epoch_idx)
            if epoch_idx % self.opt.train.save_freq == 0:
                utils.network_utils.save_checkpoints(self.ckpt_dir / Path('ckpt-epoch-%04d.pth.tar' % (epoch_idx)), \
                                                        epoch_idx, self.deblurnet, self.deblurnet_solver)
                        
            utils.network_utils.save_checkpoints(os.path.join(self.ckpt_dir, 'latest-ckpt.pth.tar'), \
                                                        epoch_idx, self.deblurnet, self.deblurnet_solver)
            
            if epoch_idx % self.opt.eval.valid_freq == 0:
                # Validation for each dataset list
                visualize_dir = self.visualize_dir / Path('epoch-' + str(epoch_idx).zfill(4))
                self.evaluation.eval_all_dataset(self.deblurnet, visualize_dir, epoch_idx)
        
            # if epoch_idx == 2:
            #     with open('profiling_results.txt', 'w') as f:
            #         f.write(prof.key_averages(group_by_stack_n=2).table(sort_by="cuda_time_total", row_limit=10))
            #     prof.export_chrome_trace("./trace.json")
            #     exit()
            
        # Close SummaryWriter for TensorBoard
        if isinstance(self.tb_writer, SummaryWriter):
            self.tb_writer.close()
        del self.evaluation