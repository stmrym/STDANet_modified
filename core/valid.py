from torch import gt
import torch.backends.cudnn
import torch.utils.data
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import cv2
import numpy as np
from utils import log

from time import time
from utils.util import ssim_calculate
from tqdm import tqdm

from models.submodules import warp

def valid(cfg, 
        val_dataset_name,
        epoch_idx, init_epoch,
        Best_Img_PSNR, Best_Epoch,
        ckpt_dir, save_dir,
        val_loader, val_transforms, deblurnet,
        tb_writer):
    
    # Set up data loader
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_loader.get_dataset(transforms = val_transforms),
        batch_size=cfg.CONST.VAL_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    inference_time = utils.network_utils.AverageMeter()
    process_time   = utils.network_utils.AverageMeter()
    img_PSNRs_mid = utils.network_utils.AverageMeter()
    img_PSNRs_out = utils.network_utils.AverageMeter()
    img_ssims_mid = utils.network_utils.AverageMeter()
    img_ssims_out = utils.network_utils.AverageMeter()

    losses_dict_list = []
    for loss_config_dict in cfg.LOSS_DICT_LIST:
        losses_dict = loss_config_dict.copy()
        losses_dict['avg_meter'] = utils.network_utils.AverageMeter()
        losses_dict_list.append(losses_dict)

    total_losses = utils.network_utils.AverageMeter()    

    deblurnet.eval()

    if epoch_idx == init_epoch:
        total_case_num = int(len(val_data_loader))
        assert total_case_num != 0, f'[{val_dataset_name}] empty!'

    tqdm_val = tqdm(val_data_loader)
    tqdm_val.set_description(f'[VALID] [Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}]')
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_val):
        
        seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
        
        torch.cuda.synchronize()
        process_start_time = time()

        with torch.no_grad():
            input_seq = []
            gt_seq = []
            input_seq += seq_blur
            input_seq = torch.cat(input_seq,1)
            gt_seq = torch.cat(seq_clear,1)
            b,t,c,h,w = gt_seq.shape

            torch.cuda.synchronize()
            inference_start_time = time()

            # Inference
            output_dict = deblurnet(input_seq) # {'recons_1': first output, 'recons_2': second output, 'recons_3': third output, 'out': final output, 'flow_forwards': fowards_list, 'flow_backwards': backwards_list}
            
            torch.cuda.synchronize()
            inference_time.update((time() - inference_start_time))

            # calculate test loss
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.VAL_BATCH_SIZE)
        
            img_PSNR_out = util.calc_psnr(output_dict['out'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_out.update(img_PSNR_out, cfg.CONST.VAL_BATCH_SIZE)
            img_PSNR_mid = util.calc_psnr(output_dict['recons_2'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_mid.update(img_PSNR_mid, cfg.CONST.VAL_BATCH_SIZE)

            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            
            tqdm_val.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')
            
            recons_2, out, flow_forwards = output_dict['recons_2'], output_dict['out'], output_dict['flow_forwards']

            if (epoch_idx % cfg.VAL.VISUALIZE_FREQ == 0):
                # saving images
                output_image = out.cpu().detach()*255
                gt_image = gt_seq[:,2,:,:,:].cpu().detach()*255

                output_image = output_image[0].permute(1,2,0)
                gt_image = gt_image[0].permute(1,2,0)

                output_image_it1 = recons_2.cpu().detach()*255
                output_image_it1 = output_image_it1[0].permute(1,2,0)
                img_ssims_mid.update(ssim_calculate(output_image_it1.numpy(),gt_image.numpy()),cfg.CONST.VAL_BATCH_SIZE)
                img_ssims_out.update(ssim_calculate(output_image.numpy(),gt_image.numpy()),cfg.CONST.VAL_BATCH_SIZE)
                seq, img_name = name[0].split('.')  # name = ['000.00000002']

                # saving output image
                if os.path.isdir(os.path.join(save_dir + '_output', seq)) == False:
                    os.makedirs(os.path.join(save_dir + '_output', seq), exist_ok=True)

                output_image = output_image.numpy().copy()
                output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(save_dir + '_output', seq, img_name + '.png'), output_image_bgr)

                for loss_dict in cfg.LOSS_DICT_LIST:
                    if 'motion_edge_loss' in loss_dict.values():
                        if os.path.isdir(os.path.join(save_dir + '_m_edge_out', seq)) == False:
                            os.makedirs(os.path.join(save_dir + '_m_edge_out', seq), exist_ok=True)
                        if os.path.isdir(os.path.join(save_dir + '_m_edge_gt', seq)) == False:
                            os.makedirs(os.path.join(save_dir + '_m_edge_gt', seq), exist_ok=True)

                        util.save_edge(
                            savename = os.path.join(save_dir + '_m_edge_out', seq, img_name + '.png'), 
                            out_image = out,
                            flow_tensor=output_dict['flow_forwards'][-1][:,1,:,:,:],
                            key = 'weighted',
                            edge_extraction_func = motion_weighted_edge_extraction)
                    
                        util.save_edge(
                            savename = os.path.join(save_dir + '_m_edge_gt', seq, img_name + '.png'),
                            out_image = gt_seq[:,2,:,:,:],
                            flow_tensor = output_dict['flow_forwards'][-1][:,1,:,:,:],
                            key = 'weighted',
                            edge_extraction_func = motion_weighted_edge_extraction)
                    
                    if 'orthogonal_edge_loss' in loss_dict.values():
                        if os.path.isdir(os.path.join(save_dir + '_o_edge_out', seq)) == False:
                            os.makedirs(os.path.join(save_dir + '_o_edge_out', seq), exist_ok=True)
                        if os.path.isdir(os.path.join(save_dir + '_o_edge_gt', seq)) == False:
                            os.makedirs(os.path.join(save_dir + '_o_edge_gt', seq), exist_ok=True)

                        util.save_edge(
                            savename = os.path.join(save_dir + '_o_edge_out', seq, img_name + '.png'), 
                            out_image = out,
                            flow_tensor=output_dict['flow_forwards'][-1][:,1,:,:,:],
                            key = 'abs_weight',
                            edge_extraction_func = orthogonal_edge_extraction)
                    
                        util.save_edge(
                            savename = os.path.join(save_dir + '_o_edge_gt', seq, img_name + '.png'),
                            out_image = gt_seq[:,2,:,:,:],
                            flow_tensor = output_dict['flow_forwards'][-1][:,1,:,:,:],
                            key = 'abs_weight',
                            edge_extraction_func = orthogonal_edge_extraction)
                        

                if cfg.VAL.SAVE_FLOW == True:
                    # saving out flow
                    out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()  
                    util.save_hsv_flow(save_dir=save_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)
                        
            
    # Output val results
    
    # Add testing results to TensorBoard
    for losses_dict in losses_dict_list:
        tb_writer.add_scalar(f'Loss_VALID_{val_dataset_name}/{losses_dict["name"]}', losses_dict["avg_meter"].avg, epoch_idx)

    tb_writer.add_scalar(f'Loss_VALID_{val_dataset_name}/TotalLoss', total_losses.avg, epoch_idx)
    tb_writer.add_scalar(f'PSNR/VALID_{val_dataset_name}', img_PSNRs_out.avg, epoch_idx)

    if img_PSNRs_out.avg  >= Best_Img_PSNR:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        Best_Img_PSNR = img_PSNRs_out.avg
        Best_Epoch = epoch_idx

    log.info(f'[VALID][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{val_dataset_name}] PSNR_mid: {img_PSNRs_mid.avg}, PSNR_out: {img_PSNRs_out.avg}, PSNR_best: {Best_Img_PSNR} at epoch {Best_Epoch}')
    log.info(f'[VALID][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{val_dataset_name}] Inference time: {inference_time}, Process time: {process_time} SSIM_mid: {img_ssims_mid.avg}, SSIM_out: {img_ssims_out.avg}')

    return Best_Img_PSNR, Best_Epoch