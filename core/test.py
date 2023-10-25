from torch import gt
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multi_loss import *
from utils import util
import torchvision
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

import numpy as np
import scipy.io as io
from utils import log

from time import time
# from visualizer import get_local
# from utils.imgio_gen import visulize_attention_ratio
from utils.util import ssim_calculate
from tqdm import tqdm
import pandas as pd
# from mmflow.datasets import visualize_flow, write_flow
from models.submodules import warp
def warp_loss(frames_list,flow_forwards,flow_backwards):
    n, t, c, h, w = frames_list.size()
    
    forward_loss = 0
    backward_loss = 0
    for flag,idx in enumerate([[1,2,3]]):
        frames = frames_list[:,idx,:,:,:]
        # for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
        flow_forward = flow_forwards
        flow_backward = flow_backwards
        # flow_forward = torch.zeros_like(flow_forwards)
        # flow_backward = torch.zeros_like(flow_backwards)
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
        forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
        forward_loss += l1Loss(forward_frames,frames_1)
        backward_loss += l1Loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)

def warp_loss_train(frames_list,flow_forwards,flow_backwards):  # copied from train.py
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
    mesh_flow[mag < minlength] = np.nan  # minlength以下をnanに置換

    u = mesh_flow[..., 0]
    v = mesh_flow[..., 1]

    return x, y, u, v

def adjust_ang(ang_min, ang_max):
    """Parameters
    input
    ang_min: start angle of degree
    ang_max: end angle of degree
    output
    unique_ang_min: angle after conversion to unique `ang_min`
    unique_ang_max: angle after conversion to unique `ang_max`
    """
    unique_ang_min = ang_min
    unique_ang_max = ang_max
    unique_ang_min %= 360
    unique_ang_max %= 360
    if unique_ang_min >= unique_ang_max:
        unique_ang_max += 360
    return unique_ang_min, unique_ang_max

def any_angle_only(mag, ang, ang_min, ang_max):
    """
    input
    mag: `cv2.cartToPolar` method `mag` reuslts
    ang: `cv2.cartToPolar` method `ang` reuslts
    ang_min: start angle of degree after `adjust_ang` function
    ang_max: end angle of degree after `adjust_ang` function
    output
    any_mag: array of replace any out of range `ang` with nan
    any_ang: array of replace any out of range `mag` with nan
    description
    Replace any out of range `mag` and `ang` with nan.
    """
    any_mag = np.copy(mag)
    any_ang = np.copy(ang)
    ang_min %= 360
    ang_max %= 360
    if ang_min < ang_max:
        any_mag[(ang < ang_min) | (ang_max < ang)] = np.nan
        any_ang[(ang < ang_min) | (ang_max < ang)] = np.nan
    else:
        any_mag[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[ang <= ang_max] += 360
    return any_mag, any_ang

def hsv_cmap(ang_min, ang_max, size):
    """
    input
    ang_min: start angle of degree after `adjust_ang` function
    ang_max: end angle of degree after `adjust_ang` function
    size: map px size
    output
    hsv_cmap_rgb: HSV color map in radial vector flow
    x, y, u, v: radial vector flow value
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    description
    Create a normalized hsv colormap between `ang_min` and `ang_max`.
    """
    # 放射状に広がるベクトル場の生成
    half = size // 2
    x = np.arange(-half, half+1, 1, dtype=np.float64)
    y = np.arange(-half, half+1, 1, dtype=np.float64)
    u, v = np.meshgrid(x, y)

    # HSV色空間の配列に入れる
    hsv = np.zeros((len(y), len(x), 3), dtype='uint8')
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    any_mag, any_ang = any_angle_only(mag, ang, ang_min, ang_max)
    hsv[..., 0] = 180*(any_ang - ang_min) / (ang_max - ang_min)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(any_mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv_cmap_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return hsv_cmap_rgb, x, y, u, v


def test(cfg, dir_dataset_name, epoch_idx, Best_Img_PSNR,ckpt_dir,dataset_loader, test_transforms, deblurnet, deblurnet_solver,test_writer):
    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)
   
    # test_data_loader = dataset_loader.loader_test
    # seq_num = len(test_data_loader)
    # Batch average meterics
    # batch_time = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs_iter2 = utils.network_utils.AverageMeter()
    img_ssims_iter1 = utils.network_utils.AverageMeter()
    img_ssims_iter2 = utils.network_utils.AverageMeter()
    deblur_mse_losses = utils.network_utils.AverageMeter()  # added for writing test loss
    warp_mse_losses = utils.network_utils.AverageMeter()    # added for writing test loss
    deblur_losses = utils.network_utils.AverageMeter()      # added for writing test loss
    warp_mse_losses_iter1 = utils.network_utils.AverageMeter()
    warp_mse_losses_iter2 = utils.network_utils.AverageMeter()
    # img_PSNRs_mid = utils.network_utils.AverageMeter()
    img_PSNRs_iter1 = utils.network_utils.AverageMeter()
    batch_end_time = time()
    # test_psnr = dict()
    # g_names= 'init'
    deblurnet.eval()
    tqdm_test = tqdm(test_data_loader)
    tqdm_test.set_description('[TEST] [Epoch {0}/{1}]'.format(epoch_idx,cfg.TRAIN.NUM_EPOCHES))
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_test):
        data_time.update(time() - batch_end_time)

        seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
        # seq_len = len(seq_blur)
        # Switch models to training mode


        # if name[0] == "IMG_0055.00077":
        
        with torch.no_grad():
            input_seq = []
            gt_seq = []
            # input_seq = [seq_blur[0] for i in range((cfg.DATA.FRAME_LENGTH-1)//2)]
            input_seq += seq_blur
            # input_seq += [seq_blur[-1] for i in range((cfg.DATA.FRAME_LENGTH-1)//2)]
            input_seq = torch.cat(input_seq,1)
            gt_seq = torch.cat(seq_clear,1)
            b,t,c,h,w = gt_seq.shape
            # np.save("sharp.npy",gt_seq.data.cpu().numpy())
            torch.cuda.synchronize()
            test_time_start = time()
            recons_1, recons_2, recons_3, out,flow_forwards,flow_backwards = deblurnet(input_seq)
            # output_img = torch.cat([recons_1, recons_2, recons_3, out],dim=1)
            # output_img_one,output_img = deblurnet(input_seq)
            torch.cuda.synchronize()
            test_time.update((time() - test_time_start)/t)

            # calculate test loss
            output_img = torch.cat([recons_1, recons_2, recons_3, out],dim=1)
            
            down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
            
            warploss = warp_loss_train(down_simple_gt, flow_forwards, flow_backwards)*0.05 

            
            warp_mse_losses.update(warploss.item(), cfg.CONST.TEST_BATCH_SIZE)

            t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)
            deblur_mse_loss = l1Loss(output_img, t_gt_seq)
            deblur_mse_losses.update(deblur_mse_loss.item(), cfg.CONST.TEST_BATCH_SIZE)

            deblur_loss = deblur_mse_loss + warploss  
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TEST_BATCH_SIZE)

            """ down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
            warploss = warp_loss(down_simple_gt, flow_forwards[1], flow_backwards[1])
            warp_mse_losses_iter1.update(warploss.item(), cfg.CONST.TEST_BATCH_SIZE)
            warploss = warp_loss(down_simple_gt, flow_forwards[-1], flow_backwards[-1])
            warp_mse_losses_iter2.update(warploss.item(), cfg.CONST.TEST_BATCH_SIZE) """
            # t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)
            # img_PSNR =  PSNR(output_img[:,-1,:,:,:], t_gt_seq[:,-1,:,:,:])
            # img_PSNR =  util.calc_psnr(output_img[:,-1,:,:,:], t_gt_seq[:,-1,:,:,:])
            # img_PSNR_tt =  PSNR(output_img, t_gt_seq)
            img_PSNR2 = util.calc_psnr(out.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter2.update(img_PSNR2, cfg.CONST.TEST_BATCH_SIZE)
            # img_PSNR = PSNR(output_img[:,:-1,:,:,:].contiguous().view(b*3,c,h,w), t_gt_seq[:,:-1,:,:,:].contiguous().view(b*3,c,h,w))
            # img_PSNR = util.calc_psnr()
            img_PSNR = util.calc_psnr(recons_2.detach(),gt_seq[:,2,:,:,:].detach())
            # img_PSNR_tt2 = PSNR(output_img[:,:-1,:,:,:].contiguous().view(b,3*c,h,w), t_gt_seq[:,:-1,:,:,:].contiguous().view(b,3*c,h,w))
            img_PSNRs_iter1.update(img_PSNR, cfg.CONST.TEST_BATCH_SIZE)
            batch_end_time = time()
            
            # log.info('[TEST] [Ech {0}/{1}][Seq {2} {3}/{4}] RT {5} DT {6}\t imgPSNR_iter1 {7} imgPSNR {8}'
                        # .format(epoch_idx, cfg.TRAIN.NUM_EPOCHES, name, seq_idx+1, seq_num, test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2))
            # log.info("[TEST] [{0} {1}]".format(img_PSNRs_iter1,img_PSNRs_iter2))
            
            # cfg.NETWORK.PHASE == 'test':
            if cfg.NETWORK.PHASE == 'test':
                output_image = out.cpu().detach()*255
                gt_image = gt_seq[:,2,:,:,:].cpu().detach()*255
                output_image = output_image[0].permute(1,2,0)
                gt_image = gt_image[0].permute(1,2,0)
                
                output_image_it1 = recons_2.cpu().detach()*255
                output_image_it1 = output_image_it1[0].permute(1,2,0)
                img_ssims_iter1.update(ssim_calculate(output_image_it1.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)
                img_ssims_iter2.update(ssim_calculate(output_image.numpy(),gt_image.numpy()),cfg.CONST.TEST_BATCH_SIZE)
                tqdm_test.set_postfix_str('RT {0} DT {1} imgPSNR_iter1 {2} imgPSNR_iter2 {3} ssim_it1 {4} ssim_it2 {5}'
                        .format(test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2,img_ssims_iter1,img_ssims_iter2))
                
                # saving images
                out_dir = os.path.join(cfg.DIR.OUT_PATH,"test",cfg.NETWORK.DEBLURNETARCH + "_" + dir_dataset_name + '_' + re.split('[/.]', cfg.CONST.WEIGHTS)[-3])
                seq, img_name = name[0].split('.')  # name = ['000.00000002']

                # saving output image
                if os.path.isdir(os.path.join(out_dir, 'output', seq)) == False:
                    os.makedirs(os.path.join(out_dir, 'output', seq), exist_ok=True)

                output_image = np.maximum(output_image.numpy().copy(), 0)
                output_image_bgr = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(out_dir, 'output', seq, img_name + '.png'), output_image_bgr)


                out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()
                H, W, _ = out_flow_forward.shape
                out_flow_forward = cv2.resize(out_flow_forward, (W*4, H*4))

                # saving flow_hsv
                # 角度範囲のパラメータ
                ang_min = 0
                ang_max = 360
                _ang_min, _ang_max = adjust_ang(ang_min, ang_max)  # 角度の表現を統一する

                # HSV色空間の配列に入れる
                hsv = np.zeros_like(output_image.astype(np.uint8), dtype='uint8')
                mag, ang = cv2.cartToPolar(out_flow_forward[..., 0], out_flow_forward[..., 1], angleInDegrees=True)
                any_mag, any_ang = any_angle_only(mag, ang, ang_min, ang_max)
                hsv[..., 0] = 180*(any_ang - _ang_min) / (_ang_max - _ang_min)
                hsv[..., 1] = 255
                # hsv[..., 2] = cv2.normalize(any_mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv[..., 2] = np.clip(any_mag * 20, 0, 255) # 正規化なし
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                # 画像の原点にHSV色空間を埋め込み
                flow_rgb_display = np.copy(flow_rgb)
                hsv_cmap_rgb, *_ = hsv_cmap(_ang_min, _ang_max, 51)
                flow_rgb_display[0:hsv_cmap_rgb.shape[0], 0:hsv_cmap_rgb.shape[1]] = hsv_cmap_rgb

                fig, ax = plt.subplots(figsize=(8, 6), dpi=350)
                ax.axis("off")
                ax.imshow(flow_rgb_display)


                forward_hsv_path = os.path.join(out_dir, 'flow_hsv', seq)
                if os.path.isdir(forward_hsv_path) == False:
                    os.makedirs(forward_hsv_path)
                plt.savefig(os.path.join(forward_hsv_path, img_name + '.png'), bbox_inches = "tight")

                plt.clf()
                plt.close()
                
                # saving flow_vector
                fig, ax = plt.subplots(figsize=(8,6), dpi=350)
                ax.axis("off")
                ax.imshow(output_image.astype(np.uint8))
                
                x, y, u, v = flow_vector(flow=out_flow_forward, spacing=10, margin=0, minlength=1)  # flow.shape must be (H, W, 2)
                im = ax.quiver(x, y, u/np.sqrt(pow(u,2)+pow(v,2)),v/np.sqrt(pow(u,2)+pow(v,2)),np.sqrt(pow(u,2)+pow(v,2)), cmap='jet', angles='xy', scale_units='xy', scale=0.1)
                
                divider = make_axes_locatable(ax) #axに紐付いたAxesDividerを取得
                cax = divider.append_axes("right", size="5%", pad=0.1) #append_axesで新しいaxesを作成
                cb = fig.colorbar(im, cax=cax)
                cb.mappable.set_clim(0.0, 8.0)


                forward_path = os.path.join(out_dir, 'flow_forward', seq)
                if os.path.isdir(forward_path) == False:
                    os.makedirs(forward_path)
                plt.savefig(os.path.join(forward_path, img_name + '.png'), bbox_inches = "tight")

                plt.clf()
                plt.close()


            else:
                tqdm_test.set_postfix_str('RT {0} DT {1} imgPSNR_iter1 {2} imgPSNR_iter2 {3}'
                        .format(test_time ,data_time, img_PSNRs_iter1,img_PSNRs_iter2))     
            
            
    # Output testing results
    if cfg.NETWORK.PHASE == 'test':

        log.info('============================ TEST RESULTS ============================')
        log.info('[TEST] Total_Mean_PSNR:itr1:{0},itr2:{1},best:{2},ssim_it1 {3},ssim_it2 {4}'.format(img_PSNRs_iter1.avg,img_PSNRs_iter2.avg,Best_Img_PSNR,img_ssims_iter1.avg,img_ssims_iter2.avg))
        
    '''
    else:
        # Output val results
        log.info('============================ TEST RESULTS ============================')
        
        

        # Add testing results to TensorBoard
        test_writer.add_scalar('Loss/EpochWarpMSELoss_TEST', warp_mse_losses.avg, epoch_idx)
        test_writer.add_scalar('Loss/EpochMSELoss_TEST', deblur_mse_losses.avg, epoch_idx)
        test_writer.add_scalar('Loss/EpochDeblurLoss_TEST', deblur_mse_losses.avg, epoch_idx) 
        test_writer.add_scalar('PSNR/Epoch_PSNR_TEST', img_PSNRs_iter2.avg, epoch_idx)
        test_writer.add_scalar('SSIM/Epoch_SSIM_TEST', img_ssims_iter2.avg, epoch_idx)
        if img_PSNRs_iter2.avg  >= Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNRs_iter2.avg
            Best_Epoch = epoch_idx
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                      epoch_idx, deblurnet,deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)
        log.info('[TEST] Total_Mean_PSNR:itr1:{0},itr2:{1},best:{2}'.format(img_PSNRs_iter1.avg,img_PSNRs_iter2.avg,Best_Img_PSNR))
        
        # test_writer.add_scalar(cfg.NETWORK.DEBLURNETARCH + '/EpochPSNR_TEST', img_PSNRs_mid.avg, epoch_idx + 1)
        return img_PSNRs_iter2.avg,Best_Img_PSNR
    '''