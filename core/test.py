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

    # test_data_loader = dataset_loader.loader_test
    # seq_num = len(test_data_loader)
    # Batch average meterics
    # batch_time = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs_iter1 = utils.network_utils.AverageMeter()
    img_PSNRs_iter2 = utils.network_utils.AverageMeter()
    img_ssims_iter1 = utils.network_utils.AverageMeter()
    img_ssims_iter2 = utils.network_utils.AverageMeter()
    deblur_mse_losses = utils.network_utils.AverageMeter()  # added for writing test loss
    warp_mse_losses = utils.network_utils.AverageMeter()    # added for writing test loss
    deblur_losses = utils.network_utils.AverageMeter()      # added for writing test loss
    # warp_mse_losses_iter1 = utils.network_utils.AverageMeter()
    # warp_mse_losses_iter2 = utils.network_utils.AverageMeter()
    # # img_PSNRs_mid = utils.network_utils.AverageMeter()

    batch_end_time = time()
    # test_psnr = dict()
    # g_names= 'init'
    deblurnet.eval()
    
    total_case_num = int(len(test_data_loader)) * cfg.CONST.TEST_BATCH_SIZE
    print(f'Total [{test_dataset_name}] test case: {total_case_num}')
    log.info(f'Total [{test_dataset_name}] test case: {total_case_num}')
    assert total_case_num != 0, f'[{test_dataset_name}] empty!'

    tqdm_test = tqdm(test_data_loader)
    tqdm_test.set_description('[TEST] [Epoch {0}/{1}]'.format(epoch_idx,cfg.TRAIN.NUM_EPOCHES))
    
    for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_test):
        data_time.update(time() - batch_end_time)

        seq_blur = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
        
        with torch.no_grad():
            input_seq = []
            gt_seq = []
            input_seq += seq_blur
            input_seq = torch.cat(input_seq,1)
            gt_seq = torch.cat(seq_clear,1)
            b,t,c,h,w = gt_seq.shape

            torch.cuda.synchronize()
            test_time_start = time()
            recons_1, recons_2, recons_3, out,flow_forwards,flow_backwards = deblurnet(input_seq)
            # output_img = torch.cat([recons_1, recons_2, recons_3, out],dim=1)
            # output_img_one,output_img = deblurnet(input_seq)
            torch.cuda.synchronize()
            test_time.update(time() - test_time_start)

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

            img_PSNR2 = util.calc_psnr(out.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter2.update(img_PSNR2, cfg.CONST.TEST_BATCH_SIZE)
            img_PSNR = util.calc_psnr(recons_2.detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_iter1.update(img_PSNR, cfg.CONST.TEST_BATCH_SIZE)
            batch_end_time = time()

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
            seq, img_name = name[0].split('.')  # name = ['000.00000002']

            # saving output image
            if os.path.isdir(os.path.join(out_dir, 'output', seq)) == False:
                os.makedirs(os.path.join(out_dir, 'output', seq), exist_ok=True)

            output_image = output_image.numpy().copy()
            output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(out_dir, 'output', seq, img_name + '.png'), output_image_bgr)

            # saving npy files
            if os.path.isdir(os.path.join(out_dir, 'flow_npy', seq)) == False:
                os.makedirs(os.path.join(out_dir, 'flow_npy', seq), exist_ok=True)
            out_flow_forward = (flow_forwards[-1])[0][1].permute(1,2,0).cpu().detach().numpy()               
            np.save(os.path.join(out_dir, 'flow_npy', seq, img_name + '.npy'),out_flow_forward)
            
    # Output testing results

    log.info('============================ TEST RESULTS ============================')
    log.info('[TEST] Total_Mean_PSNR:itr1:{0},itr2:{1},best:{2},ssim_it1 {3},ssim_it2 {4}, test_time:{5}'.format(img_PSNRs_iter1.avg,img_PSNRs_iter2.avg,Best_Img_PSNR,img_ssims_iter1.avg,img_ssims_iter2.avg, test_time.avg))

    
    # creating flow map from npy    
    log.info('========================== SAVING FLOW MAP ===========================')
    
    seqs = sorted([f for f in os.listdir(os.path.join(out_dir, 'flow_npy')) if os.path.isdir(os.path.join(out_dir, 'flow_npy', f))])

    for seq in tqdm(seqs):
        npy_files = sorted(glob.glob(os.path.join(out_dir, 'flow_npy', seq, '*.npy')))
        out_flows = []
        names = []
        for npy_file in npy_files:
            npy = np.load(npy_file)
            H, W, _ = npy.shape
            npy = cv2.resize(npy, (W*4, H*4))
            out_flows.append(npy)
            names.append(os.path.splitext((os.path.basename(npy_file)))[0])

        firstLoop = True
        for out_flow in out_flows:  # get vector_max for each seq       
            _, _, u, v = flow_vector(flow=out_flow, spacing=10, margin=0, minlength=1)  # flow.shape must be (H, W, 2)
            vector_mag = np.nanmax(np.sqrt(pow(u,2)+pow(v,2)))

            if firstLoop == True:
                vector_amax = vector_mag
                firstLoop = False
            else:
                if vector_amax < vector_mag:
                    vector_amax = vector_mag
        

        for img_name, out_flow in zip(names, out_flows):
            
            ############################
            # saving flow_hsv using mmcv
            ############################

            flow_map = visualize_flow(out_flow, None)
            # visualize_flow return flow map with RGB order
            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)

            if os.path.isdir(os.path.join(out_dir, 'flow_hsv', seq)) == False:
                os.makedirs(os.path.join(out_dir, 'flow_hsv', seq), exist_ok=True)
            
            cv2.imwrite(os.path.join(out_dir, 'flow_hsv', seq, img_name + '.png'), flow_map)            


            ####################
            # saving flow_vector
            ####################

            fig, ax = plt.subplots(figsize=(8,6), dpi=350)
            ax.axis("off")
            output_image = cv2.imread(os.path.join(out_dir, 'output', seq, img_name + '.png'), cv2.IMREAD_GRAYSCALE)

            ax.imshow(output_image.astype(np.uint8),cmap='gray', alpha=0.8)
            
            x, y, u, v = flow_vector(flow=out_flow, spacing=10, margin=0, minlength=5)  # flow.shape must be (H, W, 2)
            im = ax.quiver(x, y, u/np.sqrt(pow(u,2)+pow(v,2)),v/np.sqrt(pow(u,2)+pow(v,2)),np.sqrt(pow(u,2)+pow(v,2)), cmap='jet', angles='xy', scale_units='xy', scale=0.1)
            
            divider = make_axes_locatable(ax) # get AxesDivider
            cax = divider.append_axes("right", size="5%", pad=0.1) # make new axes
            cb = fig.colorbar(im, cax=cax)
            cb.mappable.set_clim(0, vector_amax)

            forward_path = os.path.join(out_dir, 'flow_forward', seq)
            if os.path.isdir(forward_path) == False:
                os.makedirs(forward_path)
            plt.savefig(os.path.join(forward_path, img_name + '.png'), bbox_inches = "tight")

            plt.clf()
            plt.close()
                
