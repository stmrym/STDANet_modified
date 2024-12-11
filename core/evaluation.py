from collections import OrderedDict
import torch.backends.cudnn
import torch.utils.data
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from tensorboardX import SummaryWriter
from pathlib import Path
from core.train import Trainer
import cv2
import json
import numpy as np
from time import time
from utils import log
import pandas as pd
from tqdm import tqdm
import importlib

class Evaluation(Trainer):
    def __init__(self, opt, output_dir, tb_writer=None):
        self.opt = opt
        self.output_dir = output_dir

        if hasattr(opt, 'train') and hasattr(opt.train, 'n_epochs'):
            self.n_epochs = opt.train.n_epochs
        else:
            self.n_epochs = 0    

        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.eval_transforms = self._build_transform(opt.eval_transform)
        if opt.phase == 'test':
            self.dataloader_dict = self._build_dataloader_dict(opt.dataset.test, 'test', self.eval_transforms, opt.eval_batch_size)
            self.opt.eval.visualize_freq = 1
        elif opt.phase in ['train', 'resume']:
            self.dataloader_dict = self._build_dataloader_dict(opt.dataset.val, 'valid', self.eval_transforms, opt.eval_batch_size)

        self.tb_writer = tb_writer

        # Build metric_dict
        self.metric_dict = self._set_metrics(opt.eval.metrics)
        self.inference_time = utils.network_utils.AverageMeter()
        self.process_time   = utils.network_utils.AverageMeter()
        # Batch average meterics
        self.losses_dict = self.opt.loss.copy()
        for loss_dict in self.losses_dict.values():
            loss_dict.avg_meter = utils.network_utils.AverageMeter()
        self.total_losses = utils.network_utils.AverageMeter()

    def _build_dataloader_dict(self, dataset_opt, phase, transforms, batch_size):
        # Setup datasets
        dataloader_dict = {}
        for name, dataset_dict in dataset_opt.items():
            dataset_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(input_length=self.opt.network.n_sequence, **dataset_dict)
            dataset = dataset_loader.get_dataset(transforms=transforms)
            log.info(f'[EVAL] Total {phase} case: {len(dataset)}')       
            eval_dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=os.cpu_count()//torch.cuda.device_count(), pin_memory=True, shuffle=False)
            dataloader_dict[name] = eval_dataloader
        return dataloader_dict

    def _set_metrics(self, metric_opt):
        metric_dict = {}
        for name, args in metric_opt.items():

            # module = importlib.import_module('metrics.' + name)
            module = importlib.import_module('metrics_pyiqa.' + name)
            metric_class = getattr(module, name)
            # metric_dict[name] = (metric_class(**args) if args is not None else metric_class())
            metric_dict[name] = (metric_class(self.device, **args) if args is not None else metric_class(self.device))

        return metric_dict

    def _init_evaluation(self):
        # for metric in self.metric_dict.values():
        #     metric.reset()
        self.inference_time.reset()
        self.process_time.reset()
        for loss_dict in self.losses_dict.values():
            loss_dict.avg_meter.reset()
        self.total_losses.reset()


    def _quantize(self, result_dict, keys):
        '''
        [0,1] float32 -> [0,255] uint8 -> [0,1] float32
        '''
        for k, v in result_dict.items():
            if k in keys:
                tensor_quantized = torch.clamp(v*255, 0, 255).to(torch.uint8)
                tensor_quantized = tensor_quantized.to(torch.float32)/255.0
                result_dict[k] = tensor_quantized

        return result_dict

    def _to_ndarray(self, output_dict):
        output_ndarray_dict = {}
        for k, v in output_dict.items():
            if v.dim() == 5:
                output_ndarray_dict[k] = v.detach().cpu().permute(0,1,3,4,2).numpy()*255
            if v.dim() == 4:
                output_ndarray_dict[k] = v.detach().cpu().permute(0,2,3,1).numpy()*255
        return output_ndarray_dict


    def _write_log(self, result_dict, dataset_name, epoch_idx):
        if isinstance(self.tb_writer, SummaryWriter):
            for loss_name, losses_dict in self.losses_dict.items():
                self.tb_writer.add_scalar(f'Loss_VALID_{dataset_name}/{loss_name}', losses_dict.avg_meter.avg, epoch_idx)
            self.tb_writer.add_scalar(f'Loss_VALID_{dataset_name}/TotalLoss', self.total_losses.avg, epoch_idx)


            for metric_name, seq_result in result_dict.items():
                self.tb_writer.add_scalar(f'{metric_name}/VALID_{dataset_name}', seq_result['TotalAverage'], epoch_idx)
                # self.tb_writer.add_scalar(f'{metric_name}/VALID_{dataset_name}', metric.avg, epoch_idx)

        # log_str = ' '.join([f'{key}: {value.avg:.3f}' for key, value in self.metric_dict.items()])
        log_str = ' '.join([f'{metric_name}: {seq_result["TotalAverage"]:.3f}' for metric_name, seq_result in result_dict.items()])
        log_str += f' Infer. time:{self.inference_time.avg:.3f}'
        log_str += f' Process time:{self.process_time.avg:.3f}'
        log.info(f'[EVAL][Epoch {epoch_idx}/{self.n_epochs}][{dataset_name}] ' + log_str)


    @staticmethod
    def _save_from_tensor(img_tensor, save_seq_dir, img_name):
        '''
        img_tensor: torch.Tensor [0,1] (RGB) with shape (C,H,W)
        '''
        os.makedirs(save_seq_dir, exist_ok=True)
        img_rgb = img_tensor.detach().cpu().permute(1,2,0).numpy()*255
        img_bgr = cv2.cvtColor(np.clip(img_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(os.path.join(save_seq_dir, img_name + '.png'), img_bgr)
        if not success:
            print(f'Failed to save {os.path.join(save_seq_dir, img_name + ".png")}')
            sys.exit(1)


    @staticmethod
    def _save_rgb_image(output_image, save_seq_dir, img_name):
        # saving output image
        os.makedirs(save_seq_dir, exist_ok=True)
        output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_seq_dir, img_name + '.png'), output_image_bgr)


    @staticmethod
    def _calc_avg_result(result_dict):
        for metric_name in result_dict.keys():
            # Calculating average of each seq
            for seq in list(result_dict[metric_name].keys()):
                result_dict[metric_name].setdefault('Average', {})[seq] = np.mean(list(result_dict[metric_name][seq].values()))

            # Calculating total average
            result_dict[metric_name]['TotalAverage'] = np.mean(list(result_dict[metric_name]['Average'].values()))
        
        return result_dict


    @staticmethod
    def _save_json_result(result_dict, json_save_name):
        with open(json_save_name, 'w') as f:
            json.dump(result_dict, f, indent=4, sort_keys=True)


    def _save_feat_grid(self, feat: torch.Tensor, save_name: str, nrow: int = 1) -> None:
        # feat: (N, H, W)
        # sums = feat.sum(dim=(-2,-1))
        # sorted_feat = feat[torch.argsort(sums)]
        feat = feat.unsqueeze(dim=1)
        
        # Normalize to [0, 1]
        # sorted_feat = (sorted_feat - sorted_feat.min())/(sorted_feat.max() - sorted_feat.min())
        
        # Scaling [-1, 1] -> [0, 1]
        feat = 0.5*(feat + 1)

        feat_img = torchvision.utils.make_grid(torch.clamp(feat, min=0, max=1), nrow=nrow, padding=2, normalize=False)
        torchvision.utils.save_image(feat_img, f'{save_name}.png')
        # torchvision.utils.save_image(feat, f'{save_name}')


    def _evaluation(self, deblurnet, output_dir, epoch_idx, dataset_name, dataloader):
        # Set up data loader
        result_dict = OrderedDict()
        self._init_evaluation()
        deblurnet.eval()

        visualize_dir = output_dir / 'visualization' / Path('epoch-' + str(epoch_idx).zfill(4) + '_' + dataset_name)
        json_save_name = output_dir / Path('epoch-' + str(epoch_idx).zfill(4) + '_' + dataset_name + '.json')


        tqdm_eval = tqdm(dataloader)
        tqdm_eval.set_description(f'[EVAL] [Epoch {epoch_idx}/{self.n_epochs}]')

        for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_eval):
            
            # seq_blur: [(B,C,H,W), (B,C,H,W), (B,C,H,W)]

            # name: GT frame name (center frame name)
            seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
            seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]

            with torch.no_grad():
                results = dict(
                    input=torch.cat(seq_blur,1),
                    gt=torch.cat(seq_clear,1)
                )

                
                torch.cuda.synchronize()
                inference_start_time = time()

                # Inference
                results.update(deblurnet(results['input']))

                # 'input'         : (B,T,C,H,W)
                # 'gt'            : (B,T,C,H,W)
                # 'pre_input'     : (B,T,C,H,W)
                # 'out'           : (B,C,H,W)
                # 'flow_forwards' : (B,2,2,H,W)
                # 'flow_backwards': (B,2,2,H,W)

                torch.cuda.synchronize()
                self.inference_time.update((time() - inference_start_time))
                
                # calculate test loss
                torch.cuda.synchronize()
                process_start_time = time()
                
                _ = self._calc_update_losses(results, results['gt'], self.opt.eval_batch_size)

                results = self._quantize(results, keys=['input', 'gt', 'out'])
                # results = self._to_ndarray(results)

                for batch in range(0, results['out'].shape[0]):
                    seq, img_name = name[batch].split('.')  # name = ['000.00000002']

                    recons_tensor = results['out'][batch, :,:,:]
                    gt_tensor = results['gt'][batch, results['gt'].shape[1]//2, :,:,:]    
                    lq_tensor = results['input'][batch, results['input'].shape[1]//2, :,:,:]    

                    for metric_name, metric in self.metric_dict.items():
                        result = metric.calculate(
                            recons = recons_tensor.unsqueeze(0),
                            gt = gt_tensor.unsqueeze(0),
                            lq = lq_tensor.unsqueeze(0)
                            )
                        # metric.update(result, self.opt.eval_batch_size)
                        result_dict.setdefault(metric_name, {}).setdefault(seq, {})[img_name + '.png'] = result


                    if (epoch_idx % self.opt.eval.visualize_freq == 0):
                        if self.opt.eval.save_output_img:
                            save_seq_dir = Path(str(visualize_dir) + '_output') / seq
                            self._save_from_tensor(recons_tensor, save_seq_dir, img_name)
                            # self._save_rgb_image(output_image, save_seq_dir, img_name)


                        if self.opt.eval.save_input_img:
                            preinput_tensor = results['pre_input'][batch,results['pre_input'].shape[1]//2,:,:,:]
                            save_seq_dir = Path(str(visualize_dir) + '_input') / seq
                            self._save_from_tensor(preinput_tensor, save_seq_dir, img_name)
                
                torch.cuda.synchronize()
                self.process_time.update((time() - process_start_time))
                tqdm_eval.set_postfix_str(f'Inference Time {self.inference_time.avg:.3f} Process Time {self.process_time.avg:.3f}')

        # Calc Average and save results to json
        result_dict = self._calc_avg_result(result_dict)
        self._save_json_result(result_dict, json_save_name)
        # Add testing results to TensorBoard
        self._write_log(result_dict, dataset_name, epoch_idx)

    def eval_all_dataset(self, deblurnet, output_dir, epoch_idx):
        for dataset_name, dataloader in self.dataloader_dict.items():
            self._evaluation(deblurnet, output_dir, epoch_idx, dataset_name, dataloader)


    def __del__(self):
        if hasattr(self, 'tb_writer') and isinstance(self.tb_writer, SummaryWriter):
            self.tb_writer.close()