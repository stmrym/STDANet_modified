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

def warp_loss(frames_list,flow_forwards,flow_backwards):  # copied from train.py
    n, t, c, h, w = frames_list.size()
    
    forward_loss = 0
    backward_loss = 0
    for idx in [[0,1,2],[1,2,3],[2,3,4],[1,2,3]]:
        frames = frames_list[:,idx,:,:,:]
        for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
            frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
            forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
            forward_loss += l1Loss(forward_frames,frames_1)
            backward_loss += l1Loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)/len(flow_forwards)

def valid(cfg, 
        val_dataset_name,
        epoch_idx, init_epoch,
        Best_Img_PSNR,
        ckpt_dir, save_dir,
        val_loader, val_transforms, deblurnet,
        val_writer, val_visualize):
    
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
        total_case_num = int(len(val_data_loader)) * cfg.CONST.VAL_BATCH_SIZE
        log.info(f'[VALID] Total [{val_dataset_name}] valid case: {total_case_num}')
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
            output_dict = deblurnet(input_seq) # {'recons_1': first output, 'recons_2': second output, 'recons_3': third output, 'out': final output, 'flow_fowards': fowards_list, 'flow_backwards': backwards_list}
            
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
            
            recons_2, out, flow_forwards = output_dict['recons_2'], output_dict['out'], output_dict['flow_fowards']

            if val_visualize == True:
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
                if os.path.isdir(os.path.join(save_dir, 'output', seq)) == False:
                    os.makedirs(os.path.join(save_dir, 'output', seq), exist_ok=True)

                output_image = output_image.numpy().copy()
                output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(save_dir, 'output', seq, img_name + '.png'), output_image_bgr)

                # saving mid flow npy files
                if os.path.isdir(os.path.join(save_dir, 'mid_flow_npy', seq)) == False:
                    os.makedirs(os.path.join(save_dir, 'mid_flow_npy', seq), exist_ok=True)
                mid_flow_forward = (flow_forwards[1])[0][1].permute(1,2,0).cpu().detach().numpy()               
                np.save(os.path.join(save_dir, 'mid_flow_npy', seq, img_name + '.npy'),mid_flow_forward)

                # saving out flow npy files
                if os.path.isdir(os.path.join(save_dir, 'out_flow_npy', seq)) == False:
                    os.makedirs(os.path.join(save_dir, 'out_flow_npy', seq), exist_ok=True)
                out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()               
                np.save(os.path.join(save_dir, 'out_flow_npy', seq, img_name + '.npy'),out_flow_forward)
            
    # Output val results
    log.info('============================ VALID RESULTS ============================')
    
    # Add testing results to TensorBoard
    for losses_dict in losses_dict_list:
        val_writer.add_scalar(f'Loss_VALID_{val_dataset_name}/{losses_dict["name"]}', losses_dict["avg_meter"].avg, epoch_idx)

    val_writer.add_scalar('Loss_VALID/TotalLoss', total_losses.avg, epoch_idx)
    val_writer.add_scalar(f'PSNR/VALID_{val_dataset_name}', img_PSNRs_out.avg, epoch_idx)

    if img_PSNRs_out.avg  >= Best_Img_PSNR:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        Best_Img_PSNR = img_PSNRs_out.avg

    log.info(f'[VALID] Total PSNR_mid: {img_PSNRs_mid.avg}, PSNR_out: {img_PSNRs_out.avg}, PSNR_best: {Best_Img_PSNR}')
    log.info(f'[VALID] Total SSIM_mid: {img_ssims_mid.avg}, SSIM_out: {img_ssims_out.avg}, Inference time: {inference_time}, Process time: {process_time}')

        # Creating flow map from npy    
    log.info('========================== SAVING FLOW MAP ===========================')
    
    if cfg.VAL.SAVE_FLOW == True and val_visualize == True:
        util.save_hsv_flow(save_dir=save_dir, flow_type='mid_flow', save_vector_map=False)
        util.save_hsv_flow(save_dir=save_dir, flow_type='out_flow', save_vector_map=False)

    return img_PSNRs_out.avg, Best_Img_PSNR