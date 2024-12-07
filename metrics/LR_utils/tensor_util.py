import cv2
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

from stop_watch import stop_watch

# @stop_watch
def img2tensor(img, device):
    '''
    ndarray (BGR) with shape(H, W, C) -> tensor (RGB) with shape (1, C, H, W)
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, device=device)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def tensor2img(img_tensor):
    '''
    tensor (RGB) with shape (1, C, H, W) -> ndarray (BGR) with shape (H, W, C)
    '''
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def tensor_rgb2gray(img_tensor):
    '''
    tensor (RGB) with shape (B, C, H, W) -> tensor (Gray) with shape (..., H, W)
    '''
    weights = torch.tensor([0.299, 0.5870, 0.1140], device=img_tensor.device).reshape(3,1,1)

    # weights = torch.tensor([0.299, 0.5870, 0.1140], device=img_tensor.device)

    gray_tensor = (img_tensor * weights).sum(dim=-3)
    # gray_tensor = torch.tensordot(img_tensor, weights, dims=([-3], [0]))

    return gray_tensor