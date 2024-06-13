import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Variable
from models.ops.functions import MSDeformAttnFunction
import warnings

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


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

    def _warp_w_mask(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x.device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border',align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        # mask = torch.ones(x.size(), device=x.device, requires_grad=False)
        mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output
    
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
        flow_forward02 = flow_forward01 + self._warp_w_mask(flow_forward12,flow_forward01)

        flow_backward10 = flow_backward[:,0]
        flow_backward21 = flow_backward[:,1]
        flow_backward20 = flow_backward21 + self._warp_w_mask(flow_backward10,flow_backward21)

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
    
    def process(self, query, value, sampling_offsets, attention_weights, reference_points, input_spatial_shapes, input_level_start_index, input_padding_mask, flow_forward, flow_backward):

        bs,t,c,h,w = value.shape
        # (B,T,C,H,W) -> (B,THW,C)
        query = query.flatten(3).transpose(2, 3).contiguous().view(bs,-1,c)
        value = value.flatten(3).transpose(2, 3).contiguous().view(bs,-1,c)
        
        # (B,T,2MLK,H,W) -> (B,THW,2MLK) or (B,2MLK,H,W) -> (B,HW,2MLK)
        sampling_offsets = sampling_offsets.flatten(-2).transpose(-2, -1).contiguous().view(bs,-1,self.n_heads * self.n_levels * self.n_points * 2)  
        # (B,T,MLK,H,W) -> (B,THW,MLK) or (B,MLK,H,W) -> (B,HW,MLK)
        attention_weights = attention_weights.flatten(-2).transpose(-2, -1).contiguous().view(bs,-1,self.n_heads*self.n_levels * self.n_points)
        
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = sampling_offsets.view(N, -1, self.n_heads, self.n_levels, self.n_points, 2)
        sampling_offsets = self.flow_guid_offset(flow_forward,flow_backward,sampling_offsets)
        attention_weights = attention_weights.view(N, -1, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, -1, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        


        return value, sampling_locations, sampling_offsets, attention_weights


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

        value = self.value_proj(input_flatten.view(bs*t,c,h,w)).view(bs,t,c,h,w)
        # (B,T,2MLK,H,W)
        sampling_offsets = self.sampling_offsets(query.view(bs*t,c,h,w)).view(bs,t,-1,h,w)
        # (B,T,MLK,H,W)
        attention_weights = self.attention_weights(query.view(bs*t,c,h,w)).view(bs,t,-1,h,w)

        value, sampling_locations, sampling_offsets, attention_weights = self.process(query, value, sampling_offsets, attention_weights,
                                                                reference_points, input_spatial_shapes, input_level_start_index,
                                                                input_padding_mask, flow_forward, flow_backward)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        # For visalization
        # (B, THW, M, d/M) -> (B, T, d, H, W)
        value = value.flatten(2).reshape(bs, t, h*w, -1).permute(0,1,3,2).reshape(bs,t,-1,h,w)    
        # (B, THW, M, L, K) -> (B, T, H*W, M*L*K) -> (B, T, M*L*K, H*W) -> (B, T, MLK, H, W)
        attention_weights = attention_weights.reshape(bs, t, h*w, self.n_heads, t, self.n_points).flatten(3,5).transpose(2,3).reshape(bs,t,-1,h,w)
        # (B, THW, M, L, K, 2) -> (B, T, H*W, M, L, K, 2) -> (B, 2, T, M, L, K, H*W) -> (B, 2, T, MLK, H, W)
        sampling_offsets = sampling_offsets.reshape(bs, t, h*w, self.n_heads, t, self.n_points, 2).permute(0,6,1,3,4,5,2).flatten(3,5).reshape(bs, 2, t, -1, h, w)

        output = output.view(bs,t,h*w,c).transpose(2, 3).contiguous().view(bs*t,c,h,w)
        output = self.output_proj(output)


        return output, value, sampling_offsets, attention_weights


class MSDeformAttn_Fusion(MSDeformAttn):
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

        return offset

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

        value = self.value_proj(input_flatten.view(bs*t,c,h,w)).view(bs,t,c,h,w)
        # (B,2MLK,H,W)
        sampling_offsets = self.sampling_offsets(query.view(bs,t*c,h,w)).view(bs,-1,h,w)
        # (B,MLK,H,W)
        attention_weights = self.attention_weights(query.view(bs,t*c,h,w)).view(bs,-1,h,w)
        
        value, sampling_locations, sampling_offsets, attention_weights = self.process(query, value, sampling_offsets, attention_weights,
                                                                reference_points, input_spatial_shapes, input_level_start_index,
                                                                input_padding_mask, flow_forward, flow_backward)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        # For visalization
        # (B, THW, M, d/M) -> (B, T, d, H, W)
        value = value.flatten(2).reshape(bs, t, h*w, -1).permute(0,1,3,2).reshape(bs,t,-1,h,w)
        # (B, HW, M, L, K) -> (B, HW, MLK) -> (B, MLK, HW) -> (B, MLK, H, W)
        attention_weights = attention_weights.flatten(2,4).transpose(1,2).reshape(bs,-1,h,w)
        # (B, HW, M, L, K, 2) -> (B, HW, MLK, 2) -> (B, 2, MLK, HW) -> (B, 2, MLK, H, W)
        sampling_offsets = sampling_offsets.flatten(2,4).permute(0,3,2,1).reshape(bs,2,-1,h,w)
        
        output = output.view(bs,h,w,c).permute(0,3,1,2)       
        output = self.output_proj(output)
        return output, value, sampling_offsets, attention_weights

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

    def _warp_wo_mask(self, x, flo):
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

        warp_fea01 = self._warp_wo_mask(frame[:,0],flow_backward[:,0])
        warp_fea21 = self._warp_wo_mask(frame[:,2],flow_forward[:,1])


        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward.reshape(b,-1,h,w),flow_backward.reshape(b,-1,h,w)],1))).reshape(b,t,c,h,w)
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))

        # add self.n_nevels argument
        spatial_shapes,valid_ratios = self.preprocess(value, self.n_levels)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes,valid_ratios,device=value.device)
        
        output, value, sampling_offsets, attention_weights = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)

        output = self.feed_forward(output)
        output = output.reshape(b,t,c,h,w) + frame
        
        tseq_encoder_0 = torch.cat([output.reshape(b*t,c,h,w),srcframe.reshape(b*t,c,h,w)],1)
        output = output.reshape(b*t,c,h,w) + self.feedforward(tseq_encoder_0)

        return output.reshape(b,t,c,h,w), srcframe, value, sampling_offsets, attention_weights
    

class DeformableAttnBlock_FUSION(DeformableAttnBlock):
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
        
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = self._warp_wo_mask(frame[:,0],flow_backward[:,0])
        warp_fea21 = self._warp_wo_mask(frame[:,2],flow_forward[:,1])

        qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward[:,1],flow_backward[:,0]],1))).reshape(b,t,c,h,w)
        
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        # add self.n_nevels argument
        spatial_shapes,valid_ratios = self.preprocess(value, self.n_levels)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1,2),valid_ratios,device=value.device)
        
        output, value, sampling_offsets, attention_weights = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        
        output = self.feed_forward(output)
        output = output.reshape(b,c,h,w) + frame[:,1]
        
        tseq_encoder_0 = torch.cat([output,srcframe[:,1]],1)
        output = output.reshape(b,c,h,w) + self.feedforward(tseq_encoder_0)
        output = self.fusion(output)
        return output, value, sampling_offsets, attention_weights

