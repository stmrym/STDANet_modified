import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from util import gradient_cuda
from compute_ncc_cuda import _convnfft_cuda

import torch
import torch.nn.functional as F

def align_cuda(image, ref, return_images=False):
    '''
    img: torch.Tensor (RGB) [0,1] with shape (B, C, H, W)
    ref: torch.Tensor (RGB) [0,1] with shape (B, C, H, W)
    '''
    device = image.device
    margin = torch.tensor(75, device=device)  

    # extract template with odd size
    template = image[..., margin:-margin, margin:-margin]
    template_size = torch.tensor(template.shape, device=device)
    template_size = template_size + (template_size % 2) - 1
    template_size = template_size[-2:]
    template = template[...,:template_size[0], :template_size[1]]

    # NCCの計算
    ref_size = ref.shape
    ncc_size = tuple(np.array(ref_size[-2:]) + template_size.cpu().detach().numpy() - 1)
    
    ncc = torch.zeros(ncc_size, device=device)

    for k in range(3):
        ncc += normxcorr2_cuda(template[0, k], ref[0, k])

    # ncc_margin = np.floor(template_size.cpu().numpy() / 2).astype(int)
    ncc_margin = torch.floor(template_size / 2).int()

    ncc = ncc[ncc_margin[0]:-ncc_margin[0], ncc_margin[1]:-ncc_margin[1]]

    # dy: row, dx: col
    dy2, dx2 = np.unravel_index(np.argmax(ncc.cpu().numpy()), ncc.shape)
    dy2 = dy2 - np.floor(template_size[0].cpu().numpy() / 2) - margin
    dx2 = dx2 - np.floor(template_size[1].cpu().numpy() / 2) - margin
    dy2, dx2 = int(dy2), int(dx2)

    max_idx = torch.argmax(ncc)
    dy = torch.div(max_idx, ncc.shape[-1], rounding_mode='floor')
    dx = max_idx % ncc.shape[-1]
    
    dy = (dy - 1 - torch.floor(template_size[0]/2) - margin).int()
    dx = (dx - 1 - torch.floor(template_size[1]/2) - margin).int()
    
    if return_images:
        if dy < 0:
            height = min(image.shape[-2] + dy, ref.shape[-2])
            image = image[..., -dy:height-dy, :]
            ref = ref[..., :height, :]
        else:
            height = min(image.shape[-2], ref.shape[-2] - dy)
            image = image[..., :height, :]
            ref = ref[..., dy:height+dy, :]
        if dx < 0:
            width = min(image.shape[-1] + dx, ref.shape[-1])
            image = image[..., :, -dx:width-dx]
            ref = ref[..., :, :width]
        else:
            width = min(image.shape[-1], ref.shape[-1] - dx)
            image = image[..., :, :width]
            ref = ref[..., :, dx:width+dx]
    else:
        return dx, dy

    return image, ref





def _shift_data(arr):
    min_arr = torch.min(arr)
    if min_arr < 0:
        arr -= min_arr
    return arr


def normxcorr2_cuda(template, image):
    '''
    template: torch.Tensor [0,1] with shape (H, W)
    image: torch.Tensor [0,1] with shape (H, W)
    '''
    if template.ndim > image.ndim or any(template.shape[i] > image.shape[i] for i in range(template.ndim)):
        raise ValueError("Template larger than image. Arguments are swapped.")


    template = _shift_data(template)
    image = _shift_data(image)


    cross_corr = _convnfft_cuda(torch.rot90(template, 2), image)

    m, n = template.shape
    mn = m * n

    local_sum_A2 = _local_sum(image ** 2, m, n)
    local_sum_A = _local_sum(image, m, n)

    diff_local_sums = local_sum_A2 - (local_sum_A ** 2) / mn
    denom_A = torch.sqrt(torch.clamp(diff_local_sums, min=0))

    denom_T = torch.sqrt(torch.tensor(mn - 1, dtype=torch.float32, device=image.device)) * torch.std(template, unbiased=True)
    denom = denom_T * denom_A
    numerator = cross_corr - local_sum_A * template.sum() / mn

    out = torch.zeros_like(numerator, device=image.device)

    tol = torch.sqrt(torch.tensor(torch.finfo(out.dtype).eps, device=image.device))
    
    i_nonzero = denom > tol
    out[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]

    out[torch.abs(out) - 1 > torch.sqrt(torch.tensor(torch.finfo(out.dtype).eps, device=image.device))] = 0
    return out

def _local_sum(A, m, n):
    B = F.pad(A, (n, n, m, m))
    s = torch.cumsum(B, dim=0)
    c = s[m:-1, :] - s[:-m - 1, :]
    s = torch.cumsum(c, dim=1)
    local_sum_A = s[:, n:-1] - s[:, :-n - 1]
    return local_sum_A



def _make_gaussian(len, device):
    r = len // 2
    assert r + r + 1 == len
    
    sigma = r / 3.0

    g = torch.exp(-(torch.arange(-r, r+1, device=device)**2 / (2*sigma**2)))
    g = g / g[r]

    return g



def grad_ring_cuda(latent, ref):
    '''
    latent: torch.Tensor (RGB) [0,1] with shape (B, C, H, W) 
    ref: torch.Tensor (RGB) [0,1] with shape (B, C, H, W)
    '''
    assert ref.shape[-3] == latent.shape[-3]

    g = _make_gaussian(15, latent.device)
    assert g.ndim == 1

    result = torch.zeros_like(latent, device=latent.device)

    tdy, tdx = gradient_cuda(ref)
    ldy, ldx = gradient_cuda(latent)
    g1 = g.reshape(1,1,1,-1)
    g1 = g1.expand(3,1,1,-1)

    rx = torch.abs(ldx) - F.conv2d(torch.abs(tdx), g1, padding='same', groups=3)
    ry = torch.abs(ldy) - F.conv2d(torch.abs(tdy), g1, padding='same', groups=3)
    rx = torch.clamp(rx, min=0)
    ry = torch.clamp(ry, min=0)
    result = torch.sqrt(rx**2 + ry**2)

    gy, gx = gradient_cuda(ref)
    g = torch.sqrt(gx**2 + gy**2)

    filter_width = max(max(latent.shape[-3:]) // 200, 1)
    filter_width += filter_width % 2 - 1
    emask = (g > 0.03).float()

    kernel = torch.ones((3,1,filter_width, filter_width), device=latent.device)
    emask = F.conv2d(emask, kernel, padding='same', groups=3)
    emask = (emask > 0).float()

    result[emask > 0] = 0.0
    return result