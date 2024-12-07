import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

from util import gradient_cuda
from stop_watch import stop_watch


@stop_watch
def _apply_linalg_svd(grad):
    print(grad.shape)
    exit()
    return torch.linalg.svd(grad, full_matrices=False)

# @stop_watch
def SVDCoherence_cuda(gmap):
    '''
    gmap: torch.Tensor (B*n, N, N)
    '''
    gx = gmap.real
    gy = gmap.imag

    # (B*n, N, N) -> (B*n, N*N)
    gxvect = gx.reshape(gx.shape[0], -1)
    gyvect = gy.reshape(gy.shape[0], -1)


    # (B*n, N*N) & (B*n, N*N) -> (B*n, N*N, 2)
    grad = torch.stack((gxvect, gyvect), dim=-1)

    # S: (B*n, 2)
    # _, S, _ = _apply_linalg_svd(grad)
    _, S, _ = torch.linalg.svd(grad, full_matrices=False)

    s1, s2 = S[:,0], S[:,1]
    
    co = torch.abs(s1 - s2) / (s1 + s2) 
    s1 = torch.abs(s1) # NaNを0に置き換え 
    if torch.isnan(co).any(): 
        co[torch.isnan(co)] = 0.0

    # (B*n, )
    return co, s1


def AnisoSetEst_cuda(img, N):
    '''
    img: torch.Tensor (Gray) [0, 255] with shape (B, H, W) 
    '''

    B, H, W = img.shape
    w, h = W//N, H//N

    # (B, H, W) -> (B, h, w, N, N) -> (B*num_patches, N, N)
    patches = img.unfold(1,N,N).unfold(2,N,N).reshape(-1,N,N)
    gx, gy = gradient_cuda(patches)
    G = gx + 1j*gy

    co, s1 = SVDCoherence_cuda(G)

    alpha = 0.001
    thresh = torch.tensor(alpha ** (1/(N**2-1)), device=img.device)
    thresh = torch.sqrt((1-thresh)/(1+thresh))

    mp = (co > thresh).float().reshape(B, h, w)
    return mp

@stop_watch
def MetricQ_cuda(img, N):
    '''
    img: torch.Tensor (Gray) [0, 255] with shape (B, H, W) 
    '''
    B, H, W = img.shape
    w, h = W//N, H//N

    # (B, H, W) -> (B, h, w, N, N) -> (B*num_patches, N, N)
    patches = img.unfold(1,N,N).unfold(2,N,N).reshape(-1,N,N)

    gx, gy = gradient_cuda(patches)
    G = gx + 1j*gy
    co, s1 = SVDCoherence_cuda(G)

    alpha = 0.001
    thresh = torch.tensor(alpha ** (1/(N**2-1)), device=img.device)
    thresh = torch.sqrt((1-thresh)/(1+thresh))

    # mp = (co > thresh).float().reshape(B, h, w)    
    # # 各パッチのスコアを計算 
    Q = torch.sum((co * s1) * (co > thresh)) / (w*h)
    


    return Q





