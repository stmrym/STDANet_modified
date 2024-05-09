import torch.backends.cudnn
import torch.utils.data
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import shutil
import cv2
import numpy as np
from time import time
from utils import log
from utils.util import ssim_calculate
# import lpips
from tqdm import tqdm

from models.submodules import warp

def evaluation(cfg, 
        eval_dataset_name,
        save_dir,
        eval_loader, eval_transforms, deblurnet,
        epoch_idx = 0, init_epoch = 0,
        Best_Img_PSNR = 0, Best_Epoch = 0,
        tb_writer = None):
    
    # Set up data loader
    eval_data_loader = torch.utils.data.DataLoader(
        dataset=eval_loader.get_dataset(transforms = eval_transforms),
        batch_size=cfg.CONST.EVAL_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    inference_time = utils.network_utils.AverageMeter()
    process_time   = utils.network_utils.AverageMeter()
    img_PSNRs_out = utils.network_utils.AverageMeter()
    img_ssims_out = utils.network_utils.AverageMeter()
    if cfg.NETWORK.USE_STACK:    
        img_PSNRs_mid = utils.network_utils.AverageMeter()
        img_ssims_mid = utils.network_utils.AverageMeter()

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
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_eval):
        # name: GT frame name (center frame name)
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
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.EVAL_BATCH_SIZE)

            img_PSNRs_out.update(util.calc_psnr(output_dict['out']['final'].detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
            
            output_ndarrays = output_dict['out']['final'].detach().cpu().permute(0,2,3,1).numpy()*255
            gt_ndarrays = gt_tensor.detach().cpu().permute(0,2,3,1).numpy()*255        
            
            if cfg.NETWORK.USE_STACK: 
                img_PSNRs_mid.update(util.calc_psnr(output_dict['out']['recons_2'].detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
                mid_ndarrays = output_dict['out']['recons_2'].detach().cpu().permute(0,2,3,1).numpy()*255


            for batch in range(0, output_ndarrays.shape[0]):
                output_ndarr = output_ndarrays[batch,:,:,:]
                gt_ndarr = gt_ndarrays[batch,:,:,:]
                # [TODO] need to modify
                img_ssims_out.update(ssim_calculate(output_ndarr, gt_ndarr), cfg.CONST.EVAL_BATCH_SIZE)
                
                if cfg.NETWORK.USE_STACK:
                    mid_ndarr = mid_ndarrays[batch,:,:,:]
                    img_ssims_mid.update(ssim_calculate(mid_ndarr, gt_ndarr), cfg.CONST.EVAL_BATCH_SIZE)

                if cfg.NETWORK.PHASE == 'test':
                    cfg.EVAL.VISUALIZE_FREQ = 1

                if (epoch_idx % cfg.EVAL.VISUALIZE_FREQ == 0):
                    # saving images
    
                    seq, img_name = name[batch].split('.')  # name = ['000.00000002']
                    # saving output image
                    # if os.path.isdir(os.path.join(save_dir + '_output', seq)) == False:
                    #     os.makedirs(os.path.join(save_dir + '_output', seq), exist_ok=True)

                    # output_image_bgr = cv2.cvtColor(np.clip(output_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)                    
                    # cv2.imwrite(os.path.join(save_dir + '_output', seq, img_name + '.png'), output_image_bgr)

                    # if cfg.NETWORK.USE_STACK:
                    #     if os.path.isdir(os.path.join(save_dir + '_mid', seq)) == False:
                    #         os.makedirs(os.path.join(save_dir + '_mid', seq), exist_ok=True)

                    #     mid_image_bgr = cv2.cvtColor(np.clip(mid_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)                    
                    #     cv2.imwrite(os.path.join(save_dir + '_mid', seq, img_name + '.png'), mid_image_bgr)

                    # if cfg.EVAL.SAVE_FLOW:
                    #     # saving out flow
                    #     out_flow_forward = (output_dict['flow_forwards']['final'])[0][1].permute(1,2,0).cpu().detach().numpy()  
                    #     util.save_hsv_flow(save_dir=save_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)

                    # if 'ortho_weight' in output_dict.keys():
                    #     ortho_weight = output_dict['ortho_weight']['final'][batch,0,:,:]
                    #     ortho_weight_ndarr = ortho_weight.detach().cpu().numpy()*255
                    #     if os.path.isdir(os.path.join(save_dir + '_orthoEdge', seq)) == False:
                    #         os.makedirs(os.path.join(save_dir + '_orthoEdge', seq), exist_ok=True)
                    #     cv2.imwrite(os.path.join(save_dir + '_orthoEdge', seq, img_name + '.png'), np.clip(ortho_weight_ndarr, 0, 255).astype(np.uint8))
                        

            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            if cfg.NETWORK.USE_STACK:
                tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')
            else:
                tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_out {img_PSNRs_out}')



    if cfg.EVAL.VISUAL_SAVE_FILE_MAX != -1:
        visualize_dir = save_dir.rstrip(save_dir.split('/')[-1])
        dirs = sorted([visualize_dir + dir for dir in os.listdir(visualize_dir)])
        remove_num = len(dirs) - cfg.EVAL.VISUAL_SAVE_FILE_MAX
        if remove_num > 0:
            for i in range(remove_num):
                shutil.rmtree(dirs[i])
                print(f'remove {dirs[i]}')
    
    # Add testing results to TensorBoard
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        for losses_dict in losses_dict_list:
            tb_writer.add_scalar(f'Loss_VALID_{eval_dataset_name}/{losses_dict["name"]}', losses_dict["avg_meter"].avg, epoch_idx)

        tb_writer.add_scalar(f'Loss_VALID_{eval_dataset_name}/TotalLoss', total_losses.avg, epoch_idx)
        tb_writer.add_scalar(f'PSNR/VALID_{eval_dataset_name}', img_PSNRs_out.avg, epoch_idx)
        tb_writer.add_scalar(f'SSIM/VALID_{eval_dataset_name}', img_ssims_out.avg, epoch_idx)
        # tb_writer.add_scalar(f'LPIPS/VALID_{eval_dataset_name}', img_LPIPSs_out.avg, epoch_idx)

        if img_PSNRs_out.avg  >= Best_Img_PSNR:

            Best_Img_PSNR = img_PSNRs_out.avg
            Best_Epoch = epoch_idx

    elif cfg.NETWORK.PHASE in ['test'] and tb_writer is not None:
        tb_writer.add_scalar(f'PSNR/VALID_{eval_dataset_name}', img_PSNRs_out.avg, epoch_idx)
        tb_writer.add_scalar(f'SSIM/VALID_{eval_dataset_name}', img_ssims_out.avg, epoch_idx)

    if cfg.NETWORK.USE_STACK:
        log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] PSNR(mid:{img_PSNRs_mid.avg}, out:{img_PSNRs_out.avg}), PSNR_best:{Best_Img_PSNR} at epoch {Best_Epoch}')
        log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] SSIM(mid:{img_ssims_mid.avg}, out:{img_ssims_out.avg}), Infer. time:{inference_time}, Process time:{process_time}')
    else:
        log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] PSNR(out:{img_PSNRs_out.avg}), PSNR_best:{Best_Img_PSNR} at epoch {Best_Epoch}')
        log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] SSIM(out:{img_ssims_out.avg}), Infer. time:{inference_time}, Process time:{process_time}')


    return Best_Img_PSNR, Best_Epoch