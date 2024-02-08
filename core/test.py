import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mmflow.datasets import visualize_flow
from kornia.filters import bilateral_blur

import numpy as np
from utils import log

from time import time
from utils.util import ssim_calculate
from tqdm import tqdm

def test(cfg, 
        test_dataset_name, out_dir,
        epoch_idx, Best_Img_PSNR, test_loader, test_transforms, deblurnet):
    
    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_loader.get_dataset(transforms = test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
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
    
    total_case_num = int(len(test_data_loader))
    print(f'[TEST] Total [{test_dataset_name}] test case: {total_case_num}')
    log.info(f'[TEST] Total [{test_dataset_name}] test case: {total_case_num}')
    assert total_case_num != 0, f'[{test_dataset_name}] empty!'

    tqdm_test = tqdm(test_data_loader)
    tqdm_test.set_description('[TEST]')
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_test):

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
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.TEST_BATCH_SIZE)

            img_PSNR_out = util.calc_psnr(output_dict['out'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_out.update(img_PSNR_out, cfg.CONST.TEST_BATCH_SIZE)
            img_PSNR_mid = util.calc_psnr(output_dict['recons_2'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_mid.update(img_PSNR_mid, cfg.CONST.TEST_BATCH_SIZE)

            # Calculating SSIM
            recons_2, out, flow_forwards = output_dict['recons_2'], output_dict['out'], output_dict['flow_forwards']

            output_image = out.cpu().detach()*255
            gt_image = gt_seq[:,2,:,:,:].cpu().detach()*255

            output_image = output_image[0].permute(1,2,0)
            gt_image = gt_image[0].permute(1,2,0)

            mid_image = recons_2.cpu().detach()*255
            mid_image = mid_image[0].permute(1,2,0)
            img_ssims_mid.update(ssim_calculate(mid_image.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)
            img_ssims_out.update(ssim_calculate(output_image.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)


            # Saving images
            seq, img_name = name[0].split('.')  # name = ['000.00000002']

            if os.path.isdir(os.path.join(out_dir, 'output', seq)) == False:
                os.makedirs(os.path.join(out_dir, 'output', seq), exist_ok=True)

            output_image = output_image.numpy().copy()
            output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(out_dir, 'output', seq, img_name + '.png'), output_image_bgr)

            for loss_dict in cfg.LOSS_DICT_LIST:
                if 'motion_edge_loss' in loss_dict.values():
                    
                    if os.path.isdir(os.path.join(out_dir, 'weighted_edge_out', seq)) == False:
                        os.makedirs(os.path.join(out_dir, 'weighted_edge_out', seq), exist_ok=True)
                    if os.path.isdir(os.path.join(out_dir, 'weighted_edge_gt', seq)) == False:
                        os.makedirs(os.path.join(out_dir, 'weighted_edge_gt', seq), exist_ok=True)
                    
                    util.save_edge(savename=os.path.join(out_dir, 'weighted_edge_out', seq, img_name + '.png'), out_image=out, flow_tensor=output_dict['flow_forwards'][-1][:,1,:,:,:], key='weighted', use_bilateral=False)
                    util.save_edge(savename=os.path.join(out_dir, 'weighted_edge_gt', seq, img_name + '.png'), out_image=gt_seq[:,2,:,:,:], flow_tensor=output_dict['flow_forwards'][-1][:,1,:,:,:], key='weighted', use_bilateral=True)


            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            
            if cfg.VAL.SAVE_FLOW == True:
                # saving out flow
                out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()  
                util.save_hsv_flow(save_dir=out_dir, seq=seq, img_name=img_name, out_flow=out_flow_forward)
            
            tqdm_test.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')
    

    # Output testing results
    log.info(f'[TEST] Total PSNR_mid: {img_PSNRs_mid.avg}, PSNR_out: {img_PSNRs_out.avg}')
    log.info(f'[TEST] Total SSIM_mid: {img_ssims_mid.avg}, SSIM_out: {img_ssims_out.avg}, Inference time: {inference_time}, Process time: {process_time}')
    
                
