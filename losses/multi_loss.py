import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.network_utils import *
from models.submodules import warp
from torchvision import transforms
from kornia.filters import bilateral_blur
from typing import Optional, Union

def mseLoss(output, target):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(output, target)

    return MSE

def warp_loss(output_dict:dict, gt_seq:torch.tensor):
    
    b,t,c,h,w = gt_seq.shape
    down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
    frames_list = down_simple_gt
    
    n, t, c, h, w = frames_list.size()
    flow_forwards = [output_dict['flow_forwards']]
    # ['recons_1', 'recons_2', 'recons_3', 'final']
    # flow_forwards = [output_dict['flow_forwards'][key] for key in output_dict['flow_forwards'].keys()]
    flow_backwards = [output_dict['flow_backwards']]
    # flow_backwards = [output_dict['flow_backwards'][key] for key in output_dict['flow_backwards'].keys()]

    forward_loss = 0
    backward_loss = 0
    forward_mse_loss = nn.L1Loss()
    backward_mse_loss = nn.L1Loss()

    if t == 3:
        idx_list = [[0,1,2]]
    elif t == 5:
        idx_list = [[0,1,2],[1,2,3],[2,3,4],[1,2,3]]

    for idx in idx_list:
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
    
    b,t,c,h,w = gt_seq.shape
    # output_imgs = torch.cat([output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']],dim=1)
    output_imgs = output_dict['out']
    if t == 3:
        t_gt_seq = torch.cat([gt_seq[:,1,:,:,:]],dim=1)
    elif t == 5:
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


def FFTLoss(output_dict:dict, gt_seq:torch.tensor):

    b,t,c,h,w = gt_seq.shape
    # output_imgs = torch.cat([output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']],dim=1)
    # output_imgs = torch.cat([output_dict['out'][key] for key in output_dict['out'].keys()], dim=1)
    output_imgs = output_dict['out']

    if t == 3:
        t_gt_seq = torch.cat([gt_seq[:,1,:,:,:]],dim=1)
    elif t == 5:
        t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)

    output_fft = torch.fft.fft2(output_imgs, dim=(-2, -1))
    output_fft = torch.stack([output_fft.real, output_fft.imag], dim=-1)
    t_gt_fft = torch.fft.fft2(t_gt_seq, dim=(-2, -1))
    t_gt_fft = torch.stack([t_gt_fft.real, t_gt_fft.imag], dim=-1)
    l1_loss = nn.L1Loss()
    l1 = l1_loss(output_fft, t_gt_fft)
    return l1


def laplacian_edge_extraction(img_tensor:torch.tensor):

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


def sobel_edge_extraction(img_tensor:torch.tensor):

    x_kernel = torch.cuda.FloatTensor(
        [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]
        )

    y_kernel = torch.cuda.FloatTensor(
        [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
        ]
        )
    
    sobel_x_kernel = x_kernel.expand(1, 1, 3, 3)
    sobel_y_kernel = y_kernel.expand(1, 1, 3, 3)

    with torch.no_grad():

        # grayscale
        gray = 0.114*img_tensor[:,0,:,:] + 0.587*img_tensor[:,1,:,:] + 0.299*img_tensor[:,2,:,:]
        gray = torch.unsqueeze(gray,1)
        
        # edge detection by convolution filter
        
        x_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same', padding_mode='reflect', bias=False)
        y_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same', padding_mode='reflect', bias=False)
        # set sobel kernel
        x_conv.weight = torch.nn.Parameter(sobel_x_kernel)
        y_conv.weight = torch.nn.Parameter(sobel_y_kernel)

        x_edge_tensor = x_conv(gray)
        y_edge_tensor = y_conv(gray)

    return x_edge_tensor, y_edge_tensor


def motion_weighted_edge_extraction(img_tensor:torch.tensor, flow_tensor:torch.tensor, use_bilateral:bool=True) -> dict:
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
    edge_tensor = laplacian_edge_extraction(img_tensor)

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



def orthogonal_edge_extraction(img_tensor:torch.tensor, flow_tensor:torch.tensor) -> dict:

    sobel_x, sobel_y = sobel_edge_extraction(img_tensor)
    
    sobel_amp = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-7)
    # [0, 1] normalized
    sobel_amp = sobel_amp / torch.max(sobel_amp)

    # edge_angle -> (B, 2, H, W)  ((B, dx, H, W) and (B, dy, H, W))
    edge_angle = torch.cat([sobel_x, sobel_y], dim=1)

    edge_h, edge_w = edge_angle[0,0,:,:].size()
    flow_h, flow_w = flow_tensor[0,0,:,:].size()
    
    # if input size is different, upsampling
    if (edge_h, edge_w) != (flow_h, flow_w):
        upsample = nn.Upsample(size=(edge_h, edge_w), mode='bilinear', align_corners = True)
        flow_tensor = upsample(flow_tensor)

        assert flow_tensor[:,0,:,:].shape == edge_angle[:,0,:,:].shape, f'flow_tensor size {flow_tensor.shape}, edge_angle size {edge_angle.shape} do not match!'


    # culculating inner product
    orthogonal_weight = torch.mul(flow_tensor[:,0,:,:], edge_angle[:,0,:,:]) + torch.mul(flow_tensor[:,1,:,:], edge_angle[:,1,:,:])
    # (B, H, W) -> (B, 1, H, W)
    orthogonal_weight = orthogonal_weight.unsqueeze(dim=1)

    abs_weight = torch.abs(orthogonal_weight)
    # [0, 1] normalized
    # if torch.max(abs_weight) != 0:
    # abs_weight = abs_weight / torch.max(abs_weight)

    orthogonal_edge = torch.mul(abs_weight, sobel_amp)

    return {'orthogonal':orthogonal_edge, 'abs_weight':abs_weight,'edge':sobel_amp}


def calc_loss_weighted_edge(output_tensor:torch.tensor, gt_tensor:torch.tensor, flow_tensor:torch.tensor):
    # calculating weighted edge loss for each output and GT
    output_dict = motion_weighted_edge_extraction(img_tensor=output_tensor, flow_tensor=flow_tensor, use_bilateral=False)
    gt_dict = motion_weighted_edge_extraction(img_tensor=gt_tensor, flow_tensor=flow_tensor, use_bilateral=True)
    # print(f'output {torch.max(output_dict["weighted"])} {torch.min(output_dict["weighted"])}')
    # print(f'gt {torch.max(gt_dict["weighted"])} {torch.min(gt_dict["weighted"])}')
    l1_loss = nn.SmoothL1Loss()
    loss = l1_loss(output_dict['weighted'], gt_dict['weighted'])
    return loss


def calc_loss_orthogonal_edge(output_tensor:torch.tensor, gt_tensor:torch.tensor, flow_tensor:torch.tensor):
    # calculating weighted edge loss for each output and GT
    output_dict = orthogonal_edge_extraction(img_tensor=output_tensor, flow_tensor=flow_tensor)
    gt_dict = orthogonal_edge_extraction(img_tensor=gt_tensor, flow_tensor=flow_tensor)
    # print(f'output {torch.max(output_dict["weighted"])} {torch.min(output_dict["weighted"])}')
    # print(f'gt {torch.max(gt_dict["weighted"])} {torch.min(gt_dict["weighted"])}')
    l1_loss = nn.SmoothL1Loss()
    loss = l1_loss(output_dict['abs_weight'], gt_dict['abs_weight'])
    return loss


def motion_edge_loss(output_dict:dict, gt_seq:torch.tensor):
    
    recons_1, recons_2, recons_3, out = output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']
    recons_1_ff, recons_2_ff, recons_3_ff, output_ff = output_dict['flow_forwards']
    recons_1_fb, recons_2_fb, recons_3_fb, output_fb = output_dict['flow_backwards']
    # ff, fb -> (B, 2, 2, H, W)

    # losses weighted by flow_forward
    loss  = calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=output_ff[:,1,:,:,:])
    loss += calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_2_ff[:,1,:,:,:])
    loss += calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_3_ff[:,0,:,:,:])

    # losses weighted by flow_backward
    loss += calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_1_fb[:,1,:,:,:])
    loss += calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=recons_2_fb[:,0,:,:,:])
    loss += calc_loss_weighted_edge(output_tensor=out, gt_tensor=gt_seq[:,2,:,:,:], flow_tensor=output_fb[:,0,:,:,:])
    
    return loss


def orthogonal_edge_loss(output_dict:dict, gt_seq:torch.tensor):
    
    b,t,c,h,w = gt_seq.shape
    # output_dict['out'] = {'recons_1', 'recons_2', 'recons_3', 'final'} 
    out = output_dict['out']['final']

    if t == 3:
        gt_center_seq = gt_seq[:,1,:,:,:]
    elif t == 5:
        gt_center_seq = gt_seq[:,2,:,:,:]

    output_ff = output_dict['flow_forwards']['final']
    output_fb = output_dict['flow_backwards']['final']
    # ff, fb -> (B, 2, 2, H, W)

    # losses weighted by flow_forward
    loss  = calc_loss_orthogonal_edge(output_tensor=out, gt_tensor=gt_center_seq, flow_tensor=output_ff[:,1,:,:,:])
    loss += calc_loss_orthogonal_edge(output_tensor=out, gt_tensor=gt_center_seq, flow_tensor=output_fb[:,0,:,:,:])
    
    return loss




