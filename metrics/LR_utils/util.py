import numpy as np
import torch

def mean_norm(v, p):
    return np.mean(np.abs(v)**p)**(1/p)


def my_sd(x, p):
    avg = np.mean(x)
    sd = np.mean(np.abs(x - avg)**p) ** (1.0/p)
    return sd


def mean_norm_cuda(v, p):
    return torch.mean(torch.abs(v)**p)**(1/p) 


def my_sd_cuda(x, p):
    avg = torch.mean(x)
    sd = torch.mean(torch.abs(x - avg)**p)**(1/p)
    return sd


# def gradient_cuda(input):
#     '''
#     input: torch.Tensor with shape (..., H, W)
#     '''
#     dx = (input[..., :, 2:] - input[..., :, :-2])/2
#     dx = F.pad(dx, (1,1,0,0), mode='replicate')

#     dy = (input[..., 2:, :] - input[..., :-2, :])/2
#     dy = F.pad(dy, (0,0,1,1), mode='replicate')

#     return dy, dx



def gradient_cuda(image):
    """
    画像の勾配を計算する関数 (MATLABの`gradient`関数の挙動に近い)
    :image: (H, W) の形状を持つグレースケール画像テンソル
    :return: dx, dy (両方とも (H, W) の形状)
    """
    # X方向(列方向)の勾配計算
    dx = torch.zeros_like(image)
    dx[..., :, 1:-1] = (image[..., :, 2:] - image[..., :, :-2]) / 2
    dx[..., :, 0] = image[..., :, 1] - image[..., :, 0]
    dx[..., :, -1] = image[..., :, -1] - image[..., :, -2]

    # Y方向(行方向)の勾配計算
    dy = torch.zeros_like(image)
    dy[..., 1:-1, :] = (image[..., 2:, :] - image[..., :-2, :]) / 2
    dy[..., 0, :] = image[..., 1, :] - image[..., 0, :]
    dy[..., -1, :] = image[..., -1, :] - image[..., -2, :]

    return dy, dx


