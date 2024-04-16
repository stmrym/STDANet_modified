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
import lpips
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
    img_PSNRs_mid = utils.network_utils.AverageMeter()
    img_PSNRs_out = utils.network_utils.AverageMeter()
    img_ssims_mid = utils.network_utils.AverageMeter()
    img_ssims_out = utils.network_utils.AverageMeter()
    # img_LPIPSs_mid = utils.network_utils.AverageMeter()
    # img_LPIPSs_out = utils.network_utils.AverageMeter()

    # loss_fn_alex = lpips.LPIPS(net='alex').cuda()

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

            torch.cuda.synchronize()
            inference_start_time = time()

            # Inference
            output_dict = deblurnet(input_seq) # {'recons_1': first output, 'recons_2': second output, 'recons_3': third output, 'out': final output, 'flow_forwards': fowards_list, 'flow_backwards': backwards_list}
            
            torch.cuda.synchronize()
            inference_time.update((time() - inference_start_time))

            # calculate test loss
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.EVAL_BATCH_SIZE)

            output_tensor = output_dict['out']
            mid_tensor = output_dict['recons_2']
            gt_tensor = gt_seq[:,2,:,:,:]

            img_PSNRs_out.update(util.calc_psnr(output_tensor.detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
            img_PSNRs_mid.update(util.calc_psnr(mid_tensor.detach(),gt_tensor.detach()), cfg.CONST.EVAL_BATCH_SIZE)
        
            # img_LPIPSs_out.update(loss_fn_alex(output_tensor, gt_tensor).mean().detach().cpu(), cfg.CONST.EVAL_BATCH_SIZE)
            # img_LPIPSs_mid.update(loss_fn_alex(mid_tensor, gt_tensor).mean().detach().cpu(), cfg.CONST.EVAL_BATCH_SIZE)
        
            output_ndarrays = output_tensor.detach().cpu().permute(0,2,3,1).numpy()*255
            mid_ndarrays = mid_tensor.detach().cpu().permute(0,2,3,1).numpy()*255
            gt_ndarrays = gt_tensor.detach().cpu().permute(0,2,3,1).numpy()*255

            for batch in range(0, output_ndarrays.shape[0]):
                output_ndarr = output_ndarrays[batch,:,:,:]
                mid_ndarr = mid_ndarrays[batch,:,:,:]
                gt_ndarr = gt_ndarrays[batch,:,:,:]

                img_ssims_out.update(ssim_calculate(output_ndarr, gt_ndarr), cfg.CONST.EVAL_BATCH_SIZE)
                img_ssims_mid.update(ssim_calculate(mid_ndarr, gt_ndarr), cfg.CONST.EVAL_BATCH_SIZE)

                if cfg.NETWORK.PHASE == 'test':
                    cfg.EVAL.VISUALIZE_FREQ = 1

                if (epoch_idx % cfg.EVAL.VISUALIZE_FREQ == 0):
                    # saving images
    
                    seq, img_name = name[batch].split('.')  # name = ['000.00000002']
                    # saving output image
                    if os.path.isdir(os.path.join(save_dir + '_output', seq)) == False:
                        os.makedirs(os.path.join(save_dir + '_output', seq), exist_ok=True)

                    output_image_bgr = cv2.cvtColor(np.clip(output_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)                    
                    cv2.imwrite(os.path.join(save_dir + '_output', seq, img_name + '.png'), output_image_bgr)

                    # if os.path.isdir(os.path.join(save_dir + '_mid', seq)) == False:
                    #     os.makedirs(os.path.join(save_dir + '_mid', seq), exist_ok=True)

                    # mid_image_bgr = cv2.cvtColor(np.clip(mid_ndarr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)                    
                    # cv2.imwrite(os.path.join(save_dir + '_mid', seq, img_name + '.png'), mid_image_bgr)

                    for loss_dict in cfg.LOSS_DICT_LIST:
                        if 'motion_edge_loss' in loss_dict.values():
                            if os.path.isdir(os.path.join(save_dir + '_m_edge_out', seq)) == False:
                                os.makedirs(os.path.join(save_dir + '_m_edge_out', seq), exist_ok=True)
                            if os.path.isdir(os.path.join(save_dir + '_m_edge_gt', seq)) == False:
                                os.makedirs(os.path.join(save_dir + '_m_edge_gt', seq), exist_ok=True)

                            util.save_edge(
                                savename = os.path.join(save_dir + '_m_edge_out', seq, img_name + '.png'), 
                                out_image = output_tensor,
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
                                out_image = output_tensor,
                                flow_tensor=output_dict['flow_forwards'][-1][:,1,:,:,:],
                                key = 'abs_weight',
                                edge_extraction_func = orthogonal_edge_extraction)
                        
                            util.save_edge(
                                savename = os.path.join(save_dir + '_o_edge_gt', seq, img_name + '.png'),
                                out_image = gt_tensor,
                                flow_tensor = output_dict['flow_forwards'][-1][:,1,:,:,:],
                                key = 'abs_weight',
                                edge_extraction_func = orthogonal_edge_extraction)
                            

                    if cfg.EVAL.SAVE_FLOW == True:
                        # saving out flow
                        out_flow_forward = (output_dict['flow_forwards'][-1])[0][1].permute(1,2,0).cpu().detach().numpy()  
                        util.save_hsv_flow(save_dir=save_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)



            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            tqdm_eval.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')


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

    log.info(f'[EVAL][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}][{eval_dataset_name}] PSNR(mid:{img_PSNRs_mid.avg}, out:{img_PSNRs_out.avg}), PSNR_best:{Best_Img_PSNR} at epoch {Best_Epoch}')
    log.info(f'[EVAL] Infer. time:{inference_time}, Process time:{process_time} SSIM(mid:{img_ssims_mid.avg}, out:{img_ssims_out.avg}))')

    return Best_Img_PSNR, Best_Epoch