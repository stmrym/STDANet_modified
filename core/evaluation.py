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

            module = importlib.import_module('metrics.' + name)
            metric_class = getattr(module, name)
            metric_dict[name] = (metric_class(**args) if args is not None else metric_class())

            # metric = getattr(utils.metrics, name)
            # metric_dict[name] = (metric(**args) if args is not None else metric())
        return metric_dict

    def _init_evaluation(self):
        for metric in self.metric_dict.values():
            metric.reset()
        self.inference_time.reset()
        self.process_time.reset()
        for loss_dict in self.losses_dict.values():
            loss_dict.avg_meter.reset()
        self.total_losses.reset()

        self.result_dict = OrderedDict()

    def _to_ndarray(self, output_dict):
        output_ndarray_dict = {}
        for k, v in output_dict.items():
            if v.dim() == 5:
                output_ndarray_dict[k] = v.detach().cpu().permute(0,1,3,4,2).numpy()*255
            if v.dim() == 4:
                output_ndarray_dict[k] = v.detach().cpu().permute(0,2,3,1).numpy()*255
        return output_ndarray_dict

    def _write_log(self, dataset_name, epoch_idx):
        if isinstance(self.tb_writer, SummaryWriter):
            for loss_name, losses_dict in self.losses_dict.items():
                self.tb_writer.add_scalar(f'Loss_VALID_{dataset_name}/{loss_name}', losses_dict.avg_meter.avg, epoch_idx)
            self.tb_writer.add_scalar(f'Loss_VALID_{dataset_name}/TotalLoss', self.total_losses.avg, epoch_idx)

            for metric_name, metric in self.metric_dict.items():
                self.tb_writer.add_scalar(f'{metric_name}/VALID_{dataset_name}', metric.avg, epoch_idx)
        
        log_str = ' '.join([f'{key}: {value.avg:.3f}' for key, value in self.metric_dict.items()])
        log_str += f' Infer. time:{self.inference_time.avg:.3f}'
        log_str += f' Process time:{self.process_time.avg:.3f}'
        log.info(f'[EVAL][Epoch {epoch_idx}/{self.n_epochs}][{dataset_name}] ' + log_str)

    def _save_rgb_image(self, output_image, save_seq_dir, img_name):
        # saving output image
        os.makedirs(save_seq_dir, exist_ok=True)
        output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_seq_dir, img_name + '.png'), output_image_bgr)

    def _calc_avg_result(self):
        for metric_name in self.result_dict.keys():
            # Calculating average of each seq
            for seq in list(self.result_dict[metric_name].keys()):
                self.result_dict[metric_name].setdefault('Average', {})[seq] = np.mean(list(self.result_dict[metric_name][seq].values()))

            # Calculating total average
            self.result_dict[metric_name]['TotalAverage'] = np.mean(list(self.result_dict[metric_name]['Average'].values()))

        
    def _save_json_result(self, json_save_name):
        with open(json_save_name, 'w') as f:
            json.dump(self.result_dict, f, indent=4, sort_keys=True)

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
        self._init_evaluation()
        deblurnet.eval()

        visualize_dir = output_dir / 'Visualization' / Path('epoch-' + str(epoch_idx).zfill(4) + '_' + dataset_name)
        json_save_name = output_dir / Path('epoch-' + str(epoch_idx).zfill(4) + '_' + dataset_name + '.json')


        tqdm_eval = tqdm(dataloader)
        tqdm_eval.set_description(f'[EVAL] [Epoch {epoch_idx}/{self.n_epochs}]')

        for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_eval):
            
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
                # {'out':           {'recons_1', 'recons_2', 'recons_3', 'final'},
                #  'flow_forwards': {'recons_1', 'recons_2', 'recons_3', 'final'},
                #  'flow_backwards':{'recons_1', 'recons_2', 'recons_3', 'final'},
                #  ...}
                results.update(deblurnet(results['input']))

                torch.cuda.synchronize()
                self.inference_time.update((time() - inference_start_time))
                
                # calculate test loss
                torch.cuda.synchronize()
                process_start_time = time()
                
                _ = self._calc_update_losses(results, results['gt'], self.opt.eval_batch_size)

                # if self.opt.eval.calc_metrics:
                #     self.img_PSNRs_out.update(util.calc_psnr(output_dict['out']['final'].detach(),gt_tensor.detach()), self.opt.eval_batch_size)
                #     self.img_LPIPSs_out.update(self.loss_fn_alex(output_dict['out']['final'], gt_tensor).mean().detach().cpu(), self.opt.eval_batch_size)
                
                results = self._to_ndarray(results)

                for batch in range(0, results['out'].shape[0]):

                    seq, img_name = name[batch].split('.')  # name = ['000.00000002']

                    output_image = results['out'][batch,:,:,:]
                    gt_image = results['gt'][batch, results['gt'].shape[1]//2,:,:,:]    
                    for metric_name, metric in self.metric_dict.items():
                        result = metric.calculate(img1=output_image, img2=gt_image)
                        metric.update(result, self.opt.eval_batch_size)
                        self.result_dict.setdefault(metric_name, {}).setdefault(seq, {})[img_name + '.png'] = result


                    if (epoch_idx % self.opt.eval.visualize_freq == 0):
                        if self.opt.eval.save_output_img:
                            save_seq_dir = Path(str(visualize_dir) + '_output') / seq
                            self._save_rgb_image(output_image, save_seq_dir, img_name)

                        if self.opt.eval.save_input_img:
                            input_image = results['pre_input'][batch,results['pre_input'].shape[1]//2,:,:,:]
                            save_seq_dir = Path(str(visualize_dir) + '_input') / seq
                            self._save_rgb_image(input_image, save_seq_dir, img_name)
                        # save_feat_grid((output_dict['first_scale_inblock']['final'])[batch,1], save_dir + f'{seq}_{img_name}_0_in_feat', nrow=4)
                        # save_feat_grid((output_dict['first_scale_encoder_first']['final'])[batch,1], save_dir + f'{seq}_{img_name}_1_en_feat', nrow=8)
                        # save_feat_grid((output_dict['first_scale_encoder_second']['final'])[batch,1], save_dir + f'{seq}_{img_name}_2_en_feat', nrow=8)
                        # save_feat_grid((output_dict['first_scale_encoder_second_out']['final'])[batch], save_dir + f'{seq}_{img_name}_3_en_out_feat', nrow=8)
                        # save_feat_grid((output_dict['first_scale_decoder_second']['final'])[batch], save_dir + f'{seq}_{img_name}_4_de_feat', nrow=8)
                        # save_feat_grid((output_dict['first_scale_decoder_first']['final'])[batch], save_dir + f'{seq}_{img_name}_5_de_feat', nrow=4)

                        # save_feat_grid((output_dict['sobel_edge']['final'])[batch,1], save_dir + f'{seq}_{img_name}_6_sobel_edge', nrow=1)
                        # save_feat_grid((output_dict['motion_orthogonal_edge']['final'])[batch], save_dir + f'{seq}_{img_name}_7_motion_orthogonal_edge', nrow=1)
                        # save_feat_grid((torch.abs(output_dict['motion_orthogonal_edge']['final']))[batch], save_dir + f'{seq}_{img_name}_8_abs_motion_orthogonal_edge', nrow=1)

                        # if self.opt.eval.save_flow:
                        #     # saving out flow

                        #     # torch.save(input_seq, save_dir + f'{seq}_{img_name}_input.pt')    
                        #     # torch.save(output_dict['first_scale_encoder_second']['final'], save_dir + f'{seq}_{img_name}_encoder2nd.pt')                 
                        #     # torch.save(output_dict['flow_forwards']['final'], save_dir + f'{seq}_{img_name}_flow_forwards.pt')
                        #     # torch.save(output_dict['flow_backwards']['final'], save_dir + f'{seq}_{img_name}_flow_backwards.pt')
                            
                        #     out_flow_forward = (output_dict['flow_forwards']['final'])[batch,1,:,:,:].permute(1,2,0).cpu().detach().numpy()  
                        #     util.save_hsv_flow(save_dir=save_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)


                        # if 'ortho_weight' in output_dict.keys():
                        #     ortho_weight = output_dict['ortho_weight']['final'][batch,0,:,:]
                        #     ortho_weight_ndarr = ortho_weight.detach().cpu().numpy()*255
                        #     if not os.path.isdir(os.path.join(save_dir + '_orthoEdge', seq)):
                        #         os.makedirs(os.path.join(save_dir + '_orthoEdge', seq), exist_ok=True)
                        #     cv2.imwrite(os.path.join(save_dir + '_orthoEdge', seq, img_name + '.png'), np.clip(ortho_weight_ndarr, 0, 255).astype(np.uint8))
                
                torch.cuda.synchronize()
                self.process_time.update((time() - process_start_time))
                tqdm_eval.set_postfix_str(f'Inference Time {self.inference_time.avg:.3f} Process Time {self.process_time.avg:.3f}')

        # Calc Average and save results to json
        self._calc_avg_result()
        self._save_json_result(json_save_name)
        # Add testing results to TensorBoard
        self._write_log(dataset_name, epoch_idx)

    def eval_all_dataset(self, deblurnet, output_dir, epoch_idx):
        for dataset_name, dataloader in self.dataloader_dict.items():
            self._evaluation(deblurnet, output_dir, epoch_idx, dataset_name, dataloader)


    def __del__(self):
        if hasattr(self, 'tb_writer') and isinstance(self.tb_writer, SummaryWriter):
            self.tb_writer.close()