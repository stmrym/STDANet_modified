import cv2
import kornia
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from skimage.transform import probabilistic_hough_line
from scipy.signal import fftconvolve
from scipy import linalg, fft as sp_fft
import sys
import os
sys.path.append(os.path.dirname(__file__))

from tensor_util import tensor2img
from util import gradient_cuda
from stop_watch import stop_watch


def compute_ncc_cuda(img, ref, margin):
    '''
    img: torch.Tensor (Gray) with shape (B, H, W)
    ref: torch.Tensor (Gray) with shape (B, H, W)
    img_margin: int
    '''
    assert len(img.shape) == 3 and img.shape[0] == 1, 'img shape must be (1, H, W)'
    template = ref[:, margin:-margin, margin:-margin]
    ncc = torch.ones(1, margin*2 + 1, margin*2 + 1, device=img.device) * 100
    ncc_abs = torch.ones(1, margin*2 + 1, margin*2 + 1, device=img.device) * 100

    img_mask = mask_lines_cuda(img)
    # ref_mask = mask_lines(ref)
    ref_mask = img_mask.clone()

    # img_mask_np = np.clip(img_mask[0].cpu().numpy()*255, 0, 255).astype(np.uint8)
    # cv2.imwrite('edge_mask.png', img_mask_np)

    t_mask = ref_mask[:, margin:-margin, margin:-margin]

    dy, dx = gradient_cuda(img)
    tdy, tdx = gradient_cuda(template)

    # exclude edges from gradient
    dx[img_mask] = 0
    dy[img_mask] = 0
    tdx[t_mask] = 0
    tdy[t_mask] = 0


    # ncc_dx = xcorr2_fft_cpu(tdx, dx)
    # ncc_dy = xcorr2_fft_cpu(tdy, dy)

    ncc_dx = xcorr2_fft_cuda(tdx, dx)
    ncc_dy = xcorr2_fft_cuda(tdy, dy)

    # サイズ指定でスライスを取得
    ncc_dx = ncc_dx[:, tdx.shape[1]-1:, tdx.shape[2]-1:]
    ncc_dy = ncc_dy[:, tdy.shape[1]-1:, tdy.shape[2]-1:]

    # 特定範囲のスライスを取得
    ncc_dx = ncc_dx[:, :margin * 2 + 1, :margin * 2 + 1]
    ncc_dy = ncc_dy[:, :margin * 2 + 1, :margin * 2 + 1]

    # 正規化
    ncc_dx = ncc_dx / ncc_dx[:, margin, margin]
    ncc_dy = ncc_dy / ncc_dy[:, margin, margin]

    # 絶対値を計算
    ncc_dx_abs = torch.abs(ncc_dx)
    ncc_dy_abs = torch.abs(ncc_dy)

    mask = ncc_dx_abs < ncc_abs

    ncc[mask] = ncc_dx[mask]
    ncc_abs[mask] = ncc_dx_abs[mask]

    mask = ncc_dy_abs < ncc_abs
    ncc[mask] = ncc_dy[mask]
    ncc_abs[mask] = ncc_dy_abs[mask]

    return ncc[0]




# @stop_watch
def mask_lines_cuda(img):
    '''
    img: torch.Tensor (Gray) with shape (B, H, W)
    
    -> mask: torch.Tensor (bool) with shape (B, H, W)    
    '''
    mask = torch.zeros_like(img, dtype=torch.bool, device=img.device)

    low_thr = img.max() * 0.1
    high_thr = img.max() * 0.3

    # (B, H, W) -> (B, 1, H, W) -> (B, H, W)
    m, e = kornia.filters.canny(img.unsqueeze(1), low_threshold=low_thr, high_threshold=high_thr)
    e = e.squeeze(1)

    filter = torch.ones((1,1,5,5), device=img.device)

    e_np = _tensor2ndarray(e[0])

    # cur_mask_l = []
    # for i in range(1):
    #     cur_mask = mask_line_cuda(e_np, i)
    #     cur_mask_l.append(cur_mask)
        
    # for cur_mask in cur_mask_l:

    #     cur_mask = _ndarray2tensor(cur_mask, img.device).unsqueeze(0)
    #     e[cur_mask] = False
        
    #     #  (1,H,W) -> (1,1,H,W)
    #     cur_mask = cur_mask.float().unsqueeze(0)
    #     cur_mask = F.conv2d(cur_mask, filter, padding=filter.shape[-1]//2)
    #     cur_mask = cur_mask.squeeze(0) > 0
    #     mask[cur_mask] = True


    cur_mask = mask_line_cuda(e_np, 0)
    cur_mask = _ndarray2tensor(cur_mask, img.device).unsqueeze(0)
    e[cur_mask] = False
    
    #  (1,H,W) -> (1,1,H,W)
    cur_mask = cur_mask.float().unsqueeze(0)
    cur_mask = F.conv2d(cur_mask, filter, padding=filter.shape[-1]//2)
    cur_mask = cur_mask.squeeze(0) > 0
    mask[cur_mask] = True


    # mask_image = np.clip(mask[0].cpu().numpy().astype(np.int8)*255, 0, 255).astype(np.uint8)
    # cv2.imwrite('debug_mask.png', mask_image)


    return mask

# @stop_watch
def mask_line_cuda(e_np, seed=0):
    '''
    e: canny edge torch.Tensor with shape (1, H, W)
    
    -> mask: torch.tensor lined_mask [0, 1] with shape (1, H, W)
    '''
    # e_np = _tensor2ndarray(e[0])

    # [((x1s, y1s), (x1e, y1e)), ((x2s, y2s), (x2e, 2ye)), ...]
    lines = probabilistic_hough_line(e_np, threshold=10, line_length=20, line_gap=8, rng=seed)
    
    mask = np.zeros(e_np.shape)

    for line in lines:
        if line is None:
            continue
        p1, p2 = line
        cv2.line(mask, p1, p2, 1, 1)
    
    mask = mask.astype(np.bool_)
    return mask


def _tensor2ndarray(tensor):
    '''
    tensor: torch.Tensor (H, W)
    '''
    assert len(tensor.shape) == 2, '_tensor2ndarray input must be (H, W)'
    ndarray = tensor.cpu().numpy()
    return ndarray


def _ndarray2tensor(ndarray, device):
    '''
    ndarray: np.array (H, W)
    '''
    assert len(ndarray.shape) == 2, '_ndarray2tensor input must be (H, W)'
    tensor = torch.tensor(ndarray, device=device)
    return tensor



def xcorr2_fft_cpu(a, b):
    '''
    a, b: (B, M, N)
    -> result: (B, M, N)
    '''
    # a と b は PyTorch テンソル
    device = a.device
    a = a[0].cpu().numpy()
    b = b[0].cpu().numpy()

    # b の共役転置を計算
    b_rot = np.rot90(np.conj(b), 2)
    # fftconvolve で畳み込みを実行
    result = fftconvolve(a, b_rot, mode='full')
    # PyTorch テンソルに変換して返す
    return torch.tensor(result, device=device).unsqueeze(0)


def xcorr2_fft_cuda(a, b):
    '''
    a, b: (B, M, N)
    -> result: (B, M, N)
    '''
    # b の共役転置を計算
    a = a[0]
    b_conj_rot = torch.rot90(b[0].conj(), 2, [0, 1])
    # fftconvolve に似た処理を PyTorch で実装
    result = _convnfft_cuda(a, b_conj_rot)
    return result.unsqueeze(0)




def _convnfft_cuda(A, B, dims=None):
    if dims is None:
        dims = list(range(A.ndim))
    if isinstance(dims, int):
        dims = [dims]

    original_sizes = [A.size(dim) + B.size(dim) - 1 for dim in dims]

    fshape = [sp_fft.next_fast_len(original_sizes[d], True) for d in dims]

    # FFT and IFFT along the specified dimensions

    spA = torch.fft.rfftn(A, fshape, dim=dims)
    spB = torch.fft.rfftn(B, fshape, dim=dims)

    ret = torch.fft.irfftn(spA * spB, fshape, dim=dims)

    fslice = tuple([slice(sz) for sz in original_sizes])
    ret = ret[fslice]

    return ret

    # # Define the length function for FFT
    # lfftfun = lambda l: 2**int(np.ceil(np.log2(l)))

    # for dim in dims:
    #     m = A.size(dim)
    #     n = B.size(dim)
    #     l = lfftfun(m + n - 1)

    #     A = torch.fft.fft(A, n=l, dim=dim)
    #     B = torch.fft.fft(B, n=l, dim=dim)

    # # Element-wise multiplication in the Fourier domain
    # A = A * B

    # # Inverse FFT
    # for dim in dims:
    #     A = torch.fft.ifft(A, dim=dim).real

    # # Truncate the result based on the shape
    # slices = [slice(None)] * A.ndim
    # for dim, size in zip(dims, original_sizes):
    #     slices[dim] = slice(0, size)

    # A = A[tuple(slices)]
    # return A
