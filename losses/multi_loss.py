import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.network_utils import *
from models.submodules import warp
from torchvision import transforms
from kornia.filters import bilateral_blur

def mseLoss(output, target):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(output, target)

    return MSE

def warp_loss(output_dict:dict, gt_seq:torch.tensor):
    
    b,t,c,h,w = gt_seq.shape
    down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
    frames_list = down_simple_gt
    
    n, t, c, h, w = frames_list.size()
    flow_forwards = output_dict['flow_forwards']
    flow_backwards = output_dict['flow_backwards']

    forward_loss = 0
    backward_loss = 0
    forward_mse_loss = nn.L1Loss()
    backward_mse_loss = nn.L1Loss()
    for idx in [[0,1,2],[1,2,3],[2,3,4],[1,2,3]]:
        frames = frames_list[:,idx,:,:,:]
        for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
            frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
            forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
            forward_loss += forward_mse_loss(forward_frames,frames_1)
            backward_loss += backward_mse_loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)/len(flow_forwards)


def l1Loss(output_dict:dict, gt_seq:torch.tensor):

    output_imgs = torch.cat([output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']],dim=1)
    t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)

    l1_loss = nn.L1Loss()
    l1 = l1_loss(output_imgs, t_gt_seq)
    return l1

def PSNR(output, target, max_val = 1.0,shave = 4):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target[:,:,shave:-shave,shave:-shave] - output[:,:,shave:-shave,shave:-shave], 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)

def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss


def edge_extraction(img_tensor:torch.tensor):

    # Laplacian filter
    kernel = torch.cuda.FloatTensor([[1, 1, 1], 
                                     [1, -8, 1], 
                                     [1, 1, 1]])
    edge_k = kernel.expand(1, 1, 3, 3)

    with torch.no_grad():

        # grayscale
        gray = 0.114*img_tensor[:,0,:,:] + 0.587*img_tensor[:,1,:,:] + 0.299*img_tensor[:,2,:,:]
        gray = torch.unsqueeze(gray,1)
        
        # edge detection by convolution filter
        edge_tensor = F.conv2d(gray, edge_k, padding='same')
    
        # Clamp to >= 0
        edge_tensor = torch.clamp(input=edge_tensor, min=0)
        # edge_tensor = (edge_tensor - torch.min(edge_tensor))/(torch.max(edge_tensor) - torch.min(edge_tensor))

    return edge_tensor


def motion_weighted_edge_extraction(img_tensor:torch.tensor, flow_tensor:torch.tensor, use_bilateral:bool) -> dict:
    #################### 
    # Input shape   
    # img_tensor -> (B, C, H, W)
    # flow_tensor -> (B, 2, H, W)
    ####################

    #  use bilateral filter to extract GT edge
    if use_bilateral == True:
        img_tensor = bilateral_blur(input=img_tensor, kernel_size=(5, 5), sigma_color=0.1, sigma_space=(1.5, 1.5))
        img_tensor = bilateral_blur(input=img_tensor, kernel_size=(5, 5), sigma_color=0.1, sigma_space=(1.5, 1.5))
    
    # edge_tensor -> (B, 1, H, W)
    edge_tensor = edge_extraction(img_tensor)

    edge_h, edge_w = edge_tensor[0,0,:,:].size()
    flow_h, flow_w = flow_tensor[0,0,:,:].size()
    
    # if input size is different, upsampling
    if (edge_h, edge_w) != (flow_h, flow_w):
        upsample = nn.Upsample(size=(edge_h, edge_w), mode='bilinear')
        flow_tensor = upsample(flow_tensor)

        assert flow_tensor[:,0,:,:].shape == edge_tensor[:,0,:,:].shape, f'flow_tensor size {flow_tensor.shape}, edge_tensor size {edge_tensor.shape} do not match!'

    # calculate flow magnitude from delta-x ([:,0,:,:]) and delta-y ([:,1,:,:])
    # flow_magnitude_tensor -> (B, 1, H, W)
    flow_magnitude_tensor = torch.sqrt(flow_tensor[:,0,:,:]**2 + flow_tensor[:,1,:,:]**2 + 1e-6).unsqueeze(dim=1)

    # element-wise product
    weighted_edge_tensor = torch.mul(edge_tensor, flow_magnitude_tensor)
    
    # (B, 1, H, W)
    return {'weighted':weighted_edge_tensor, 'edge':edge_tensor, 'flow_magnitude':flow_magnitude_tensor}


def calc_weighted_edge_loss(output_tensor:torch.tensor, gt_tensor:torch.tensor, flow_tensor:torch.tensor):
    # calculating weighted edge loss for each output and GT
    output_dict = motion_weighted_edge_extraction(img_tensor=output_tensor, flow_tensor=flow_tensor, use_bilateral=False)
    gt_dict = motion_weighted_edge_extraction(img_tensor=gt_tensor, flow_tensor=flow_tensor, use_bilateral=True)
    # print(f'output {torch.max(output_dict["weighted"])} {torch.min(output_dict["weighted"])}')
    # print(f'gt {torch.max(gt_dict["weighted"])} {torch.min(gt_dict["weighted"])}')
    l1_loss = nn.SmoothL1Loss()
    loss = l1_loss(output_dict['weighted'], gt_dict['weighted'])
    return loss


def motion_edge_loss(output_dict:dict, gt_seq:torch.tensor):
    
    recons_1, recons_2, recons_3, out = output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']
    recons_1_ff, recons_2_ff, recons_3_ff, output_ff = output_dict['flow_forwards']
    recons_1_fb, recons_2_fb, recons_3_fb, output_fb = output_dict['flow_backwards']
    # ff, fb -> (B, 2, 2, H, W)

    # losses weighted by flow_forward
    loss  = calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=output_ff[:,1,:,:,:])
    loss += calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_2_ff[:,1,:,:,:])
    loss += calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_3_ff[:,0,:,:,:])

    # losses weighted by flow_backward
    loss += calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_1_fb[:,1,:,:,:])
    loss += calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_2_fb[:,0,:,:,:])
    loss += calc_weighted_edge_loss(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=output_fb[:,0,:,:,:])
    
    return loss


def calc_update_losses(output_dict:dict, gt_seq:torch.tensor, losses_dict_list:list, total_losses, batch_size:int):

    total_loss = 0
    for losses_dict in losses_dict_list:
        loss = eval(losses_dict['func'])(output_dict, gt_seq) * losses_dict['weight']   # Calculate loss
        losses_dict['avg_meter'].update(loss.item(), batch_size)    # Update loss
        total_loss += loss

    total_losses.update(total_loss.item(), batch_size) # Update total losses
    return total_loss, total_losses, losses_dict_list


