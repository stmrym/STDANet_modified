
import torch.backends.cudnn
import torch.utils.data
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import lpips
import shutil
import cv2
import numpy as np
from time import time
from utils import log
from utils.util import ssim_calculate
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.submodules import warp

def make_eval_df(cfg, epoch_average_list: list, save_name: str) -> pd.DataFrame:
    col_name = ['seq', 'mid_SSIM', 'out_SSIM', 'mid_SSIM_sd', 'out_SSIM_sd'] if cfg.NETWORK.USE_STACK else ['seq', 'out_SSIM', 'out_SSIM_sd']
    eval_df = pd.DataFrame(epoch_average_list, columns=col_name)
    for col in col_name[1:]:
        eval_df.at['Avg.', col] = eval_df[col].mean()

    if os.path.isfile(save_name): # Add mode
        eval_df.to_csv(save_name, mode='a', index=False, header=False)
    else:   # Create csv file
        eval_df.to_csv(save_name, mode='x', index=False)
    
    return eval_df

def add_epoch_average_list(cfg, epoch_average_list: list, seq_df: pd.DataFrame) -> list:
    if cfg.NETWORK.USE_STACK:
        epoch_average_list.append([seq_df['seq'][0], seq_df['mid_SSIM'].mean(), seq_df['out_SSIM'].mean(), np.sqrt(seq_df['mid_SSIM'].var()), np.sqrt(seq_df['out_SSIM'].var())])
    else:
        epoch_average_list.append([seq_df['seq'][0], seq_df['out_SSIM'].mean(), np.sqrt(seq_df['out_SSIM'].var())])
    return epoch_average_list

def make_seq_df(cfg, seq_frame_value_list: list, epoch_average_list: list, save_dir: str, save_name: str) -> list:
    col_name = ['seq', 'frame', 'mid_SSIM', 'out_SSIM'] if cfg.NETWORK.USE_STACK else ['seq', 'frame', 'out_SSIM']
    seq_df = pd.DataFrame(seq_frame_value_list, columns=col_name)
    if not os.path.isdir(os.path.join(save_dir)):
        os.makedirs(save_dir, exist_ok=True)
    seq_df.to_csv(os.path.join(save_dir, save_name), index=False)
    epoch_average_list = add_epoch_average_list(cfg, epoch_average_list, seq_df)
    return epoch_average_list

def save_feat_grid(feat: torch.Tensor, save_name: str, nrow: int = 1) -> None:
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


def evaluation(cfg, 
        eval_dataset_name: str,
        save_dir: str,
        eval_loader: utils.data_loaders.VideoDeblurDataLoader_No_Slipt, 
        eval_transforms: utils.data_transforms.Compose, 
        deblurnet: torch.nn.DataParallel,
        epoch_idx: int = 0, 
        init_epoch: int = 0,
        Best_Img_PSNR: int = 0, 
        Best_Epoch: int = 0,
        tb_writer = None):
    
    # Set up data loader
    eval_data_loader = torch.utils.data.DataLoader(
        dataset=eval_loader.get_dataset(transforms = eval_transforms),
        batch_size=cfg.CONST.EVAL_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    inference_time = utils.network_utils.AverageMeter()
    process_time   = utils.network_utils.AverageMeter()
    if cfg.EVAL.CALC_METRICS:
        img_PSNRs_out = utils.network_utils.AverageMeter()
        img_LPIPSs_out = utils.network_utils.AverageMeter()
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    if cfg.NETWORK.USE_STACK:    
        img_PSNRs_mid = utils.network_utils.AverageMeter()
        img_LPIPSs_mid = utils.network_utils.AverageMeter()

    losses_dict_list = []
    for loss_config_dict in cfg.LOSS_DICT_LIST:
        losses_dict = loss_config_dict.copy()
        losses_dict['avg_meter'] = utils.network_utils.AverageMeter()
        losses_dict_list.append(losses_dict)

    total_losses = utils.network_utils.AverageMeter()    

    deblurnet.eval()

    if epoch_idx == init_epoch:
        total_case_num = int(len(eval_data_loader))
        assert total_case_num != 0, f'[{eval_dataset_name}] empty!'

    tqdm_eval = tqdm(eval_data_loader)
    tqdm_eval.set_description(f'[EVAL] [Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}]')

    epoch_average_list = []

    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_eval):
        
        # name: GT frame name (center frame name)
        seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]

        with torch.no_grad():
            input_seq = []
            gt_seq = []
            input_seq += seq_blur
            input_seq = torch.cat(input_seq,1)
            gt_seq = torch.cat(seq_clear,1)

            if len(seq_clear) == 3:
                gt_tensor = gt_seq[:,1,:,:,:]
            elif len(seq_clear) == 5:
                gt_tensor = gt_seq[:,2,:,:,:]
            
            torch.cuda.synchronize()
            inference_start_time = time()

            # Inference

            # {'out':           {'recons_1', 'recons_2', 'recons_3', 'final'},
            #  'flow_forwards': {'recons_1', 'recons_2', 'recons_3', 'final'},
            #  'flow_backwards':{'recons_1', 'recons_2', 'recons_3', 'final'},
            #  ...}
            output_dict = deblurnet(input_seq)

            torch.cuda.synchronize()
            inference_time.update((time() - inference_start_time))
            
            # calculate test loss
            torch.cuda.synchronize()
            process_start_time = time()
            
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.EVAL_BATCH_SIZE)

            if cfg.EVAL.CALC_METRICS:
                img_PSNRs_out.update(util.calc_psnr(output_dict['out']['final'].detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
                img_LPIPSs_out.update(loss_fn_alex(output_dict['out']['final'], gt_tensor).mean().detach().cpu(), cfg.CONST.EVAL_BATCH_SIZE)
            

            output_ndarrays = output_dict['out']['final'].detach().cpu().permute(0,2,3,1).numpy()*255
            gt_ndarrays = gt_tensor.detach().cpu().permute(0,2,3,1).numpy()*255        
            
            if cfg.NETWORK.USE_STACK:
                if cfg.EVAL.CALC_METRICS: 
                    img_PSNRs_mid.update(util.calc_psnr(output_dict['out']['recons_2'].detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
                    img_LPIPSs_mid.update(loss_fn_alex(output_dict['out']['recons_2'], gt_tensor).mean().detach().cpu(), cfg.CONST.EVAL_BATCH_SIZE)
                mid_ndarrays = output_dict['out']['recons_2'].detach().cpu().permute(0,2,3,1).numpy()*255

            for batch in range(0, output_ndarrays.shape[0]):
                
                if cfg.EVAL.CALC_METRICS:
                    if seq_idx == 0 and batch == 0:
                        # Initialize seq_frame_value_list
                        seq_frame_value_list = []
                    elif seq != name[batch].split('.')[0]:
                            # End of sequence, and make dataFrame
                            csv_savedir = save_dir +  '_csv'
                            epoch_average_list = make_seq_df(cfg, seq_frame_value_list, epoch_average_list, csv_savedir, str(epoch_idx) + '_' + seq + '.csv')
                            # Initialize for next sequence
                            seq_frame_value_list = []
                seq, img_name = name[batch].split('.')  # name = ['000.00000002']
                        
                output_ndarr = output_ndarrays[batch,:,:,:]
                gt_ndarr = gt_ndarrays[batch,:,:,:]    
                output_image_bgr = cv2.cvtColor(np.clip(output_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                gt_image_bgr = cv2.cvtColor(np.clip(gt_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                if cfg.NETWORK.USE_STACK:
                    mid_ndarr = mid_ndarrays[batch,:,:,:]
                    mid_image_bgr = cv2.cvtColor(np.clip(mid_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                if cfg.EVAL.CALC_METRICS:
                    img_ssim_out = ssim_calculate(output_image_bgr, gt_image_bgr)

                    if cfg.NETWORK.USE_STACK:
                        img_ssim_mid = ssim_calculate(mid_image_bgr, gt_image_bgr)
                        seq_frame_value_list.append([seq, int(img_name), img_ssim_mid, img_ssim_out])
                    else:
                        seq_frame_value_list.append([seq, int(img_name), img_ssim_out])

                if cfg.NETWORK.PHASE == 'test':
                    cfg.EVAL.VISUALIZE_FREQ = 1

                if (epoch_idx % cfg.EVAL.VISUALIZE_FREQ == 0):

                    # saving output image
                    if not os.path.isdir(os.path.join(save_dir + '_output', seq)):
                        os.makedirs(os.path.join(save_dir + '_output', seq), exist_ok=True)
                    
                    cv2.imwrite(os.path.join(save_dir + '_output', seq, img_name + '.png'), output_image_bgr)

                    if cfg.NETWORK.USE_STACK:
                        if not os.path.isdir(os.path.join(save_dir + '_mid', seq)):
                            os.makedirs(os.path.join(save_dir + '_mid', seq), exist_ok=True)               
                        cv2.imwrite(os.path.join(save_dir + '_mid', seq, img_name + '.png'), mid_image_bgr)


                    # save_feat_grid((output_dict['first_scale_inblock']['final'])[batch,1], save_dir + f'{seq}_{img_name}_0_in_feat', nrow=4)
                    # save_feat_grid((output_dict['first_scale_encoder_first']['final'])[batch,1], save_dir + f'{seq}_{img_name}_1_en_feat', nrow=8)
                    # save_feat_grid((output_dict['first_scale_encoder_second']['final'])[batch,1], save_dir + f'{seq}_{img_name}_2_en_feat', nrow=8)
                    # save_feat_grid((output_dict['first_scale_encoder_second_out']['final'])[batch], save_dir + f'{seq}_{img_name}_3_en_out_feat', nrow=8)
                    # save_feat_grid((output_dict['first_scale_decoder_second']['final'])[batch], save_dir + f'{seq}_{img_name}_4_de_feat', nrow=8)
                    # save_feat_grid((output_dict['first_scale_decoder_first']['final'])[batch], save_dir + f'{seq}_{img_name}_5_de_feat', nrow=4)

                    # save_feat_grid((output_dict['sobel_edge']['final'])[batch,1], save_dir + f'{seq}_{img_name}_6_sobel_edge', nrow=1)
                    # save_feat_grid((output_dict['motion_orthogonal_edge']['final'])[batch], save_dir + f'{seq}_{img_name}_7_motion_orthogonal_edge', nrow=1)
                    # save_feat_grid((torch.abs(output_dict['motion_orthogonal_edge']['final']))[batch], save_dir + f'{seq}_{img_name}_8_abs_motion_orthogonal_edge', nrow=1)
                    
                    # exit()

                    if cfg.EVAL.SAVE_FLOW:
                        # saving out flow

                        # torch.save(input_seq, save_dir + '_input.pt')                 
                        # torch.save(output_dict['flow_forwards']['final'], save_dir + '_flow_forwards.pt')
                        # torch.save(output_dict['flow_backwards']['final'], save_dir + '_flow_backwards.pt')
                        
                        out_flow_forward = (output_dict['flow_forwards']['final'])[batch,1,:,:,:].permute(1,2,0).cpu().detach().numpy()  
                        util.save_hsv_flow(save_dir=save_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)


                    if 'ortho_weight' in output_dict.keys():
                        ortho_weight = output_dict['ortho_weight']['final'][batch,0,:,:]
                        ortho_weight_ndarr = ortho_weight.detach().cpu().numpy()*255
                        if not os.path.isdir(os.path.join(save_dir + '_orthoEdge', seq)):
                            os.makedirs(os.path.join(save_dir + '_orthoEdge', seq), exist_ok=True)
                        cv2.imwrite(os.path.join(save_dir + '_orthoEdge', seq, img_name + '.png'), np.clip(ortho_weight_ndarr, 0, 255).astype(np.uint8))
            
            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            if cfg.EVAL.CALC_METRICS:
                if cfg.NETWORK.USE_STACK:
                    tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')
                else:
                    tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_out {img_PSNRs_out}')
            else:
                if cfg.NETWORK.USE_STACK:
                    tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time}')
                else:
                    tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time}')
    
    if cfg.EVAL.CALC_METRICS:
        # Make dataFrame of last sequence    
        csv_savedir = save_dir +  '_csv'            
        epoch_average_list = make_seq_df(cfg, seq_frame_value_list, epoch_average_list, csv_savedir, str(epoch_idx) + '_' + seq + '.csv')
        # Make average dataFrame at the epoch
        eval_df = make_eval_df(cfg, epoch_average_list, save_dir + '_average.csv')
    
    # Add testing results to TensorBoard
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        for losses_dict in losses_dict_list:
            tb_writer.add_scalar(f'Loss_VALID_{eval_dataset_name}/{losses_dict["name"]}', losses_dict["avg_meter"].avg, epoch_idx)

        tb_writer.add_scalar(f'Loss_VALID_{eval_dataset_name}/TotalLoss', total_losses.avg, epoch_idx)
        tb_writer.add_scalar(f'PSNR/VALID_{eval_dataset_name}', img_PSNRs_out.avg, epoch_idx)
        tb_writer.add_scalar(f'SSIM/VALID_{eval_dataset_name}', eval_df.at['Avg.', 'out_SSIM'], epoch_idx)
        tb_writer.add_scalar(f'LPIPS/VALID_{eval_dataset_name}', img_LPIPSs_out.avg, epoch_idx)

        if img_PSNRs_out.avg  >= Best_Img_PSNR:

            Best_Img_PSNR = img_PSNRs_out.avg
            Best_Epoch = epoch_idx

    elif cfg.NETWORK.PHASE in ['test'] and tb_writer is not None:
        if cfg.EVAL.CALC_METRICS:
            tb_writer.add_scalar(f'PSNR/VALID_{eval_dataset_name}', img_PSNRs_out.avg, epoch_idx)
            tb_writer.add_scalar(f'SSIM/VALID_{eval_dataset_name}', eval_df.at['Avg.', 'out_SSIM'], epoch_idx)
            tb_writer.add_scalar(f'LPIPS/VALID_{eval_dataset_name}', img_LPIPSs_out.avg, epoch_idx)

    if cfg.EVAL.CALC_METRICS:
        if cfg.NETWORK.USE_STACK:
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] PSNR(mid:{img_PSNRs_mid.avg}, out:{img_PSNRs_out.avg}), PSNR_best:{Best_Img_PSNR} at epoch {Best_Epoch}')
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] SSIM(mid:{eval_df.at["Avg.","mid_SSIM"]}, out:{eval_df.at["Avg.","out_SSIM"]}), Infer. time:{inference_time}, Process time:{process_time}')
        else:
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] PSNR(out:{img_PSNRs_out.avg}), PSNR_best:{Best_Img_PSNR} at epoch {Best_Epoch}')
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] SSIM(out:{eval_df.at["Avg.","out_SSIM"]}), Infer. time:{inference_time}, Process time:{process_time}')

    else:
        if cfg.NETWORK.USE_STACK:
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] Infer. time:{inference_time}, Process time:{process_time}')
        else:
            log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] Infer. time:{inference_time}, Process time:{process_time}')       


    return Best_Img_PSNR, Best_Epoch