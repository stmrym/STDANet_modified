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

import numpy as np
from utils import log

from time import time
from utils.util import ssim_calculate
from tqdm import tqdm
import glob
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

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def flow_vector(flow, spacing, margin, minlength):
    """Parameters:
    input
    flow: motion vectors 3D-array
    spacing: pixel spacing of the flow
    margin: pixel margins of the flow
    minlength: minimum pixels to leave as flow
    output
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    """
    h, w, _ = flow.shape

    x = np.arange(margin, w - margin, spacing, dtype=np.int64)
    y = np.arange(margin, h - margin, spacing, dtype=np.int64)

    mesh_flow = flow[np.ix_(y, x)]
    mag, _ = cv2.cartToPolar(mesh_flow[..., 0], mesh_flow[..., 1])
    mesh_flow[mag < minlength] = np.nan  # replace under minlength to nan

    u = mesh_flow[..., 0]
    v = mesh_flow[..., 1]

    return x, y, u, v

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
    
    total_case_num = int(len(test_data_loader)) * cfg.CONST.TEST_BATCH_SIZE
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
            output_dict = deblurnet(input_seq) # {'recons_1': first output, 'recons_2': second output, 'recons_3': third output, 'out': final output, 'flow_fowards': fowards_list, 'flow_backwards': backwards_list}

            torch.cuda.synchronize()
            inference_time.update((time() - inference_start_time))

            # calculate test loss
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.TEST_BATCH_SIZE)


            img_PSNR_out = util.calc_psnr(output_dict['out'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_out.update(img_PSNR_out, cfg.CONST.TEST_BATCH_SIZE)
            img_PSNR_mid = util.calc_psnr(output_dict['recons_2'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_mid.update(img_PSNR_mid, cfg.CONST.TEST_BATCH_SIZE)

            # Calculating SSIM
            recons_2, out, flow_forwards = output_dict['recons_2'], output_dict['out'], output_dict['flow_fowards']

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

            torch.cuda.synchronize()
            process_time.update((time() - process_start_time))
            
            # Saving npy files
            if os.path.isdir(os.path.join(out_dir, 'out_flow_npy', seq)) == False:
                os.makedirs(os.path.join(out_dir, 'out_flow_npy', seq), exist_ok=True)
            out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()               
            np.save(os.path.join(out_dir, 'out_flow_npy', seq, img_name + '.npy'),out_flow_forward)
            
            tqdm_test.set_postfix_str(f'Inference Time {inference_time} Process Time {process_time} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')
            

    # Output testing results
    log.info('============================ TEST RESULTS ============================')
    log.info(f'[TEST] Total PSNR_mid: {img_PSNRs_mid.avg}, PSNR_out: {img_PSNRs_out.avg}')
    log.info(f'[TEST] Total SSIM_mid: {img_ssims_mid.avg}, SSIM_out: {img_ssims_out.avg}, Inference time: {inference_time}, Process time: {process_time}')
    
    # Creating flow map from npy    
    log.info('========================== SAVING FLOW MAP ===========================')
    
    util.save_hsv_flow(out_dir, flow_type='out_flow', save_vector_map=True)
                
