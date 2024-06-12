import os
import cv2
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.ops.functions import MSDeformAttnFunction
from torch.nn.init import xavier_uniform_, constant_
from models.submodules import warp
from torchvision.utils import save_image
from typing import List, Tuple
from matplotlib import cm
from utils.save_util import save_multi_tensor

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn_Fusion(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, kernel_size=3):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.kernel_size = kernel_size

        # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        kernel_size = 3
        self.sampling_offsets = nn.Conv2d(n_levels*d_model, n_heads * n_levels * n_points * 2, kernel_size=kernel_size, padding=kernel_size//2)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Conv2d(n_levels*d_model, n_heads * n_levels * n_points, kernel_size=kernel_size, padding=kernel_size//2)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.output_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))
        # constant_(self.sampling_offsets.bias, 0.)
        # constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    def flow_guid_offset(self,flow_forward,flow_backward,offset):
        # sampling_offsets = sampling_offsets.reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # sampling_offsets: B,T*H*W,heads,T,K,2
        
        # flow:b,2,h,w ----> b,3*h*w,2
        # (N, Length_{query}, n_levels, 2)
        # reference_points[:, :, None, :, None, :]
        # N, h*w, self.n_heads, self.n_levels, self.n_points, 2
        N,HW,heads,n_levels,points,_ = offset.shape
        # N,T,HW,heads,n_levels,points,2
        # offset = offset.reshape(N,,heads,n_levels,points,2)
        # [4, 1, 4096, 8, 3, 12, 2]
        # offset_chunk0,offset_chunk1,offset_chunk2 = torch.chunk(offset, n_levels, dim=3)
        
        # 4,2,64,64
        # flow_forward01 = flow_forward[:,0]
        flow_forward12 = flow_forward[:,1]
        # flow_forward02 = flow_forward01 + warp(flow_forward12,flow_forward01)

        flow_backward10 = flow_backward[:,0]
        flow_zeros = torch.zeros_like(flow_forward12)
        flow_stack = torch.stack([flow_backward10,flow_zeros,flow_forward12],dim=2)
        # 3,2,3,64,64 
        # N,HW,n_levels,2
        offset = offset + flow_stack.reshape(N,2,n_levels,HW).permute(0,3,2,1)[:,:,None,:,None,:]
        # flow_backward21 = flow_backward[:,1]
        # flow_backward20 = flow_backward21 + warp(flow_backward10,flow_backward21)

        # b,c,h,w = flow_backward10.shape
        # 4,h*w,2
        # flow_forward01 = flow_forward01.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_forward12 = flow_forward12.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_forward02 = flow_forward02.permute(0, 2, 3, 1).reshape(b,-1,c)

        # flow_backward10 = flow_backward10.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_backward21 = flow_backward21.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_backward20 = flow_backward20.permute(0, 2, 3, 1).reshape(b,-1,c)

        # flow_zeros = torch.zeros_like(flow_forward01)
        # 4,4096,3,2
        # offset_chunk0 = offset_chunk0 + torch.stack([flow_zeros,flow_forward01,flow_forward02],dim=2)[:,None,:,None,:,None,:]
        # offset_chunk1 = offset_chunk1 + torch.stack([flow_backward10,flow_zeros,flow_forward12],dim=2)[:,None,:,None,:,None,:]
        # offset_chunk2 = offset_chunk2 + torch.stack([flow_backward20,flow_backward21,flow_zeros],dim=2)[:,None,:,None,:,None,:]

        # offset = torch.cat([offset_chunk0,offset_chunk1,offset_chunk2],dim=1).reshape( N,THW,heads,n_levels,points,2)

        return offset

    def _reset_offset(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask,flow_forward,flow_backward):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        bs,t,c,h,w = query.shape
        value = self.value_proj(input_flatten.reshape(bs*t,c,h,w)).reshape(bs,t,c,h,w)

        # (B,2MLK,H,W)
        sampling_offsets = self.sampling_offsets(query.reshape(bs,t*c,h,w)).reshape(bs,-1,h,w)
        # (B,MLK,H,W)
        attention_weights = self.attention_weights(query.reshape(bs,t*c,h,w)).reshape(bs,-1,h,w)
        

        query = query.flatten(3).transpose(2, 3).contiguous().reshape(bs,-1,c)        
        value = value.flatten(3).transpose(2, 3).contiguous().reshape(bs,-1,c)
        sampling_offsets = sampling_offsets.flatten(2).transpose(1, 2).contiguous().reshape(bs,-1,self.n_heads * self.n_levels * self.n_points * 2)
        attention_weights = attention_weights.flatten(2).transpose(1, 2).contiguous().reshape(bs,-1,self.n_heads*self.n_levels * self.n_points)
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
       
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.reshape(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = sampling_offsets.reshape(N, h*w, self.n_heads, self.n_levels, self.n_points, 2)
        sampling_offsets = self.flow_guid_offset(flow_forward,flow_backward,sampling_offsets)
        attention_weights = attention_weights.reshape(N, h*w, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).reshape(N, h*w, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # print(reference_points[:, :, None, :, None, :].shape)
            # print(offset_normalizer[None, None, None, :, None, :].shape)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
    
        # # visalization
        # # (B, THW, M, d/M) -> (B, T, d, H, W)
        # value_out = value.flatten(2).reshape(bs, t, h*w, -1).permute(0,1,3,2).reshape(bs,t,-1,h,w)
        # # (B, HW, M, L, K) -> (B, HW, MLK) -> (B, MLK, HW) -> (B, MLK, H, W)
        # attention_weights_out = attention_weights.flatten(2,4).transpose(1,2).reshape(bs,-1,h,w)
        # # (B, HW, M, L, K, 2) -> (B, HW, MLK, 2) -> (B, 2, MLK, HW) -> (B, 2, MLK, H, W)
        # sampling_offsets_out = sampling_offsets.flatten(2,4).permute(0,3,2,1).reshape(bs,2,-1,h,w)
        # base_dir = './exp_log/test/2024-06-10T105227_F_STDAN_Stack'
        
        # # base_dir = './exp_log/test/2024-06-11T091207_Mi11Lite_ESTDANv2/feat'
        # if not os.path.isdir(base_dir):
        #     os.makedirs(base_dir, exist_ok=True)
        # save_multi_tensor(value_out[0], base_dir + '/msa_value', normalize_range=[-1, 1], nrow=8, cmap=None)
        # save_multi_tensor(attention_weights_out[0], base_dir + '/msa_attention_weights', normalize_range=[0, 1], nrow=12, cmap='jet')
        # save_multi_tensor(sampling_offsets_out[0,0], base_dir + '/msa_sampling_offsets_x', normalize_range=[-40, 40], nrow=12, cmap='bwr')
        # # save_multi_tensor(sampling_offsets[0,1], base_dir + '/sampling_offsets', normalize_range=[-40, 40], nrow=12, cmap='bwr')
        # exit()

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = output.reshape(bs,h,w,c).permute(0,3,1,2)
        output = self.output_proj(output)
        return output

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, kernel_size=3):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                        "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.kernel_size = kernel_size

        # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets = nn.Conv2d(d_model, n_heads * n_levels * n_points * 2, kernel_size=kernel_size, padding=kernel_size//2)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Conv2d(d_model, n_heads * n_levels * n_points, kernel_size=kernel_size, padding=kernel_size//2)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.output_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # constant_(self.sampling_offsets.bias, 0.)
        # constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def flow_guid_offset(self,flow_forward,flow_backward,offset):
        # sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # sampling_offsets: B,T*H*W,heads,T,K,2
        
        # flow:b,2,h,w ----> b,3*h*w,2
        # (N, Length_{query}, n_levels, 2)
        # reference_points[:, :, None, :, None, :]
        N,THW,heads,n_levels,points,_ = offset.shape
        # N,T,HW,heads,n_levels,points,2
        offset = offset.reshape(N,n_levels,-1,heads,n_levels,points,2)
        # [4, 1, 4096, 8, 3, 12, 2]
        offset_chunk0,offset_chunk1,offset_chunk2 = torch.chunk(offset, n_levels, dim=1)
        
        # 4,2,64,64
        flow_forward01 = flow_forward[:,0]
        flow_forward12 = flow_forward[:,1]
        flow_forward02 = flow_forward01 + warp(flow_forward12,flow_forward01)

        flow_backward10 = flow_backward[:,0]
        flow_backward21 = flow_backward[:,1]
        flow_backward20 = flow_backward21 + warp(flow_backward10,flow_backward21)

        b,c,h,w = flow_backward10.shape
        # 4,h*w,2
        flow_forward01 = flow_forward01.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward12 = flow_forward12.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward02 = flow_forward02.permute(0, 2, 3, 1).reshape(b,-1,c)

        flow_backward10 = flow_backward10.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_backward21 = flow_backward21.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_backward20 = flow_backward20.permute(0, 2, 3, 1).reshape(b,-1,c)

        flow_zeros = torch.zeros_like(flow_forward01)
        # 4,4096,3,2
        
        offset_chunk0 = offset_chunk0 + torch.stack([flow_zeros,flow_forward01,flow_forward02],dim=2)[:,None,:,None,:,None,:]
        offset_chunk1 = offset_chunk1 + torch.stack([flow_backward10,flow_zeros,flow_forward12],dim=2)[:,None,:,None,:,None,:]
        offset_chunk2 = offset_chunk2 + torch.stack([flow_backward20,flow_backward21,flow_zeros],dim=2)[:,None,:,None,:,None,:]
        offset = torch.cat([offset_chunk0,offset_chunk1,offset_chunk2],dim=1).reshape( N,THW,heads,n_levels,points,2)        
        
        return offset
    
    def _reset_offset(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask,flow_forward,flow_backward):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        bs,t,c,h,w = query.shape

        value = self.value_proj(input_flatten.reshape(bs*t,c,h,w)).reshape(bs,t,c,h,w)
        # (B,T,2MLK,H,W)
        sampling_offsets = self.sampling_offsets(query.reshape(bs*t,c,h,w)).reshape(bs,t,-1,h,w)
        # (B,T,MLK,H,W)
        attention_weights = self.attention_weights(query.reshape(bs*t,c,h,w)).reshape(bs,t,-1,h,w)
        
        ####process


        # (B,T,C,H,W) -> (B,THW,C)
        query = query.flatten(3).transpose(2, 3).reshape(bs,-1,c)
        value = value.flatten(3).transpose(2, 3).reshape(bs,-1,c)

        # (B,THW,2MLK) or (B,HW,2MLK)
        sampling_offsets = sampling_offsets.flatten(3).transpose(2, 3).reshape(bs,-1,self.n_heads * self.n_levels * self.n_points * 2)
        # (B,THW,MLK) or (B,HW,MLK)
        attention_weights = attention_weights.flatten(3).transpose(2, 3).reshape(bs,-1,self.n_heads*self.n_levels * self.n_points)
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.reshape(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = sampling_offsets.reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        sampling_offsets = self.flow_guid_offset(flow_forward,flow_backward,sampling_offsets)
        attention_weights = attention_weights.reshape(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # print(reference_points[:, :, None, :, None, :].shape)
            # print(offset_normalizer[None, None, None, :, None, :].shape)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # # visalization
        # # (B, THW, M, d/M) -> (B, T, d, H, W)
        # value_out = value.flatten(2).reshape(bs, t, h*w, -1).permute(0,1,3,2).reshape(bs,t,-1,h,w)    
        # # (B, THW, M, L, K) -> (B, T, H*W, M*L*K) -> (B, T, M*L*K, H*W) -> (B, T, MLK, H, W)
        # attention_weights_out = attention_weights.reshape(bs, t, h*w, self.n_heads, t, self.n_points).flatten(3,5).transpose(2,3).reshape(bs,t,-1,h,w)
        # # (B, THW, M, L, K, 2) -> (B, T, H*W, M, L, K, 2) -> (B, 2, T, M, L, K, H*W) -> (B, 2, T, MLK, H, W)
        # sampling_offsets_out = sampling_offsets.reshape(bs, t, h*w, self.n_heads, t, self.n_points, 2).permute(0,6,1,3,4,5,2).flatten(3,5).reshape(bs, 2, t, -1, h, w)
        # print(attention_weights_out.shape)
        # print(sampling_offsets_out.shape)
        # # base_dir = './exp_log/test/2024-06-10T104202_F_STDAN'
        # base_dir = './exp_log/test/2024-06-10T105227_F_STDAN_Stack'
        # save_multi_tensor(value_out[0], base_dir + '/mma_value3', normalize_range=[-1, 1], nrow=8, cmap=None)
        # save_multi_tensor(attention_weights_out[0], base_dir + '/mma_attention_weights', normalize_range=[0, 1], nrow=12, cmap='jet')
        # save_multi_tensor(sampling_offsets_out[0,0], base_dir + '/mma_sampling_offsets_x', normalize_range=[-40, 40], nrow=12, cmap='bwr')
        # # save_multi_tensor(sampling_offsets[0,1], base_dir + '/sampling_offsets_y_mma', normalize_range=[-40, 40], nrow=12, cmap='bwr')


        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        ##### process_end
        output = output.view(bs,t,h*w,c).transpose(2, 3).contiguous().view(bs*t,c,h,w)
        output = self.output_proj(output)
        return output


class MSDeformAttn_Fusion_new(MSDeformAttn):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, kernel_size=3):
        super().__init__(d_model, n_levels, n_heads, n_points, kernel_size)



class DeformableAttnBlock(nn.Module):
    def __init__(self,n_heads=4,n_levels=3,n_points=4,d_model=32):
        super().__init__()
        self.n_levels = n_levels
        
        self.defor_attn = MSDeformAttn(d_model=d_model,n_levels=self.n_levels,n_heads=n_heads,n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3*d_model+8, 3*d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3*d_model, 3*d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1,inplace=True)

        
        self.feedforward = nn.Sequential(
            nn.Conv2d(2*d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            )
        self.act = nn.LeakyReLU(0.1,inplace=True)
        
        
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs, n_lvls):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(n_lvls):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = warp(frame[:,0],flow_backward[:,0])
        warp_fea21 = warp(frame[:,2],flow_forward[:,1])


        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward.reshape(b,-1,h,w),flow_backward.reshape(b,-1,h,w)],1))).reshape(b,t,c,h,w)
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        # base_dir = './exp_log/test/2024-06-10T105227_F_STDAN_Stack'
        # save_multi_tensor(value[0], base_dir + '/mma_value2', normalize_range=[-1, 1], nrow=8, cmap=None)
        # save_multi_tensor(frame[0], base_dir + '/mma_value1', normalize_range=[-1, 1], nrow=8, cmap=None)

        # add self.n_nevels argument
        spatial_shapes,valid_ratios = self.preprocess(value, self.n_levels)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes,valid_ratios,device=value.device)
        
        output = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        
        output = self.feed_forward(output)
        output = output.reshape(b,t,c,h,w) + frame
        
        tseq_encoder_0 = torch.cat([output.reshape(b*t,c,h,w),srcframe.reshape(b*t,c,h,w)],1)
        output = output.reshape(b*t,c,h,w) + self.feedforward(tseq_encoder_0)
        return output.reshape(b,t,c,h,w),srcframe
    
class DeformableAttnBlock_FUSION(nn.Module):
    def __init__(self,n_heads=4,n_levels=3,n_points=4,d_model=32):
        super().__init__()
        self.n_levels = n_levels
        
        self.defor_attn = MSDeformAttn_Fusion(d_model=d_model,n_levels=self.n_levels,n_heads=n_heads,n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3*d_model+4, 3*d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3*d_model, 3*d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1,inplace=True)

        
        self.feedforward = nn.Sequential(
            nn.Conv2d(2*d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            )
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1,inplace=True)
            )
        
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs, n_lvls):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(n_lvls):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = warp(frame[:,0],flow_backward[:,0])
        warp_fea21 = warp(frame[:,2],flow_forward[:,1])


        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward[:,1],flow_backward[:,0]],1))).reshape(b,t,c,h,w)
        
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        # add self.n_nevels argument
        spatial_shapes,valid_ratios = self.preprocess(value, self.n_levels)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1,2),valid_ratios,device=value.device)
        
        output = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        
        output = self.feed_forward(output)
        output = output.reshape(b,c,h,w) + frame[:,1]
        
        tseq_encoder_0 = torch.cat([output,srcframe[:,1]],1)
        output = output.reshape(b,c,h,w) + self.feedforward(tseq_encoder_0)
        output = self.fusion(output)
        return output
    
def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
        xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x.device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border',align_corners=True)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )

        # mask[mask < 0.999] = 0
        # mask[mask > 0] = 1

        # output = output * mask

        return output





def read_image_from_filename(filename_list: List[str]) -> List[np.ndarray]:

    image_list = []
    for filename in filename_list:
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list


def convert_input_tesnor_list(input_list: List[np.ndarray]) -> List[torch.Tensor]:
    # Convert np.array list to input tensor list

    # [np.arrray(h, w, c), ...] -> torch.Tensor(n, h, w, c)
    inputs = np.stack(input_list)
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).cuda()
    inputs = inputs.float() / 255
    # (1, n, c, h, w)
    input_tensor = torch.unsqueeze(inputs, dim = 0)

    num_frame = input_tensor.shape[1]
    input_tensor_list = []
    for i in range(0, num_frame - 4):
        input_tensor_list.append(input_tensor[:, i:i+5, :, :, :])

    return input_tensor_list


def vis_flow():

    img1 = cv2.imread('/mnt/d/results/20240529/2024-05-13T191746_STDAN_GOPRO_GOPRO_raw epoch 1200/00034.png')
    img2 = cv2.imread('./exp_log/test/debug_/flow_out_flow/001/00034.png')
    img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)

    alpha = 0.5
    img = alpha * img1.astype(np.float32) + (1-alpha) *img2.astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite('/mnt/d/results/20240529/2024-05-13T191746_STDAN_GOPRO_GOPRO_raw epoch 1200/blend.png', img)

    # filename_list = [
    #     '../dataset/chronos/test/0306-222113/sharp/00032.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00033.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00034.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00035.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00036.png'
    # ]
    # image_list = read_image_from_filename(filename_list)
    # gt_seq = convert_input_tesnor_list(image_list)[0]
    

    input_seq = torch.load('./debug_results/input.pt')
    # gt_seq = torch.load('./debug_results/gt_seq.pt')
    flow_forward = torch.load('./debug_results/flow_forwards.pt')
    flow_backward = torch.load('./debug_results/flow_backwards.pt')
    

    flow_forward_npy = flow_forward.detach().cpu().numpy()
    np.savetxt('./debug_results/forward_00.csv', flow_forward_npy[0,0,0,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_01.csv', flow_forward_npy[0,0,1,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_10.csv', flow_forward_npy[0,1,0,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_11.csv', flow_forward_npy[0,1,1,:,:], fmt='%.3f', delimiter=',')



    b,t,c,h,w = input_seq.shape
    down_simple_gt = F.interpolate(input_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
    # frames = down_simple_gt[:,[1,2,3],:,:,:]
    frames = down_simple_gt[:,[0,1,2],:,:,:]
    n, t, c, h, w = frames.size()
    
    frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
    frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)

    backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
    forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))



    backward_frames = backward_frames.reshape(b,t-1,3,h,w)
    forward_frames = forward_frames.reshape(b,t-1,3,h,w)
    save_image(backward_frames[0,0], './debug_results/gt_backword_2.png')
    save_image(backward_frames[0,1], './debug_results/gt_backword_3.png')
    save_image(forward_frames[0,0], './debug_results/gt_forword_1.png')
    save_image(forward_frames[0,1], './debug_results/gt_forword_2.png')


def get_cmap_rgb(image_tensor: torch.tensor, cmap_name: str = 'bwr') -> torch.tensor:
    colormap = cm.get_cmap(cmap_name, 256)
    converted_tensor = torch.stack([torch.tensor(colormap(x.item())[0:3]) for x in image_tensor])
    return converted_tensor


def examine_stda_module():

    device = 'cuda:0'
    # weight = './exp_log/train/2024-06-02T124807_F_ESTDAN_v2_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-1200.pth.tar'
    # weight = './exp_log/train/F_2024-05-29T122237_STDAN_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-1200.pth.tar'
    weight = './exp_log/train/2024-06-02T141809_FFTloss_added_at_E700_STDAN_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-1200.pth.tar'
    # weight = './exp_log/train/F_2024-05-31T115702_ESTDAN_v2_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-1200.pth.tar'
    checkpoint = torch.load(weight, map_location='cpu')
    base_dir = './exp_log/test/2024-06-10T105227_F_STDAN_Stack'
    # base_dir = './exp_log/test/2024-06-11T091207_Mi11Lite_ESTDANv2'
    tensor_name = 'Mi11LiteVID_20240523_164838_00067'

    first_scale_encoder_second = torch.load(os.path.join(base_dir, tensor_name + '_encoder2nd.pt')).to(device)
    flow_forward = torch.load(os.path.join(base_dir, tensor_name + '_flow_forwards.pt')).to(device)
    flow_backward = torch.load(os.path.join(base_dir, tensor_name + '_flow_backwards.pt')).to(device)
    
    # print(first_scale_encoder_second.shape)
    # print(flow_forward.shape)

    mma = DeformableAttnBlock(n_heads=4,d_model=128,n_levels=3,n_points=12).to(device)
    msa = DeformableAttnBlock_FUSION(n_heads=4,d_model=128,n_levels=3,n_points=12).to(device)

    mma.load_state_dict({k.replace('module.recons_net.MMA.', ''):v for k,v in checkpoint['deblurnet_state_dict'].items() if 'MMA' in k })
    msa.load_state_dict({k.replace('module.recons_net.MSA.', ''):v for k,v in checkpoint['deblurnet_state_dict'].items() if 'MSA' in k })

    frame,srcframe = mma(first_scale_encoder_second,first_scale_encoder_second,flow_forward,flow_backward)
    encoder_2nd_out = msa(frame,srcframe,flow_forward,flow_backward)


    # print(frame.shape)
    # print(encoder_2nd_out.shape)

    b, t, c, h, w = first_scale_encoder_second.shape

    save_multi_tensor(first_scale_encoder_second[0], base_dir + '/encoder_2nd', [-1, 1], nrow=8)
    save_multi_tensor(flow_forward.reshape(b,-1,h,w), base_dir + '/flow_forwards', [-20, 20], nrow=2, cmap_name='bwr')
    save_multi_tensor(flow_backward.reshape(b,-1,h,w), base_dir + '/flow_backwards', [-20, 20], nrow=2, cmap_name='bwr')
    save_multi_tensor(frame[0], base_dir + '/encoder_2nd_mid', [-1, 1], nrow=8) 
    save_multi_tensor(encoder_2nd_out[0], base_dir + '/encoder_2nd_out', [-1, 1], nrow=8)



if __name__ == '__main__':
    examine_stda_module()
    # tensor = torch.load('./debug_results/F_2024-05-31T115702_ESTDAN_v2_BSD_3ms24ms_GOPRO_stda/encoder_2nd.pt')
    # save_multi_tensor(tensor[0], './debug_results/F_2024-05-31T115702_ESTDAN_v2_BSD_3ms24ms_GOPRO_stda/ff.png', normalize_range = [-1, 1], nrow=8, cmap='jet')

