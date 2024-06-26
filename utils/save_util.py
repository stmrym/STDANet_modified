import numpy as np
import torch
import torchvision
from typing import List
from matplotlib import cm

def gray2cmap_tensor(image_tensor: torch.tensor, cmap_name: str = 'bwr') -> torch.tensor:
    colormap = cm.get_cmap(cmap_name, 256)
    # (N, H, W)
    image_array = image_tensor.to('cpu').detach().numpy().copy()
    # (N, H, W, 3)
    converted_array = colormap(image_array)[:,:,:,0:3]
    # (N, 3, H, W)
    converted_tensor = torch.from_numpy(converted_array).clone().permute(0,3,1,2)
    return converted_tensor

def gray2cmap_numpy(image_np: np.ndarray, cmap_name: str = 'bwr') -> np.ndarray:
    colormap = cm.get_cmap(cmap_name, 256)
    # (H, W) -> (H, W, 3)
    converted_array = colormap(image_np)[:,:,0:3]
    return converted_array

def save_feat_grid(feat: torch.Tensor, save_name: str, normalize_range: List[int] = [-1, 1], nrow: int = 1, cmap: str = None) -> None:
    # feat: (N, H, W)
    # sums = feat.sum(dim=(-2,-1))
    # sorted_feat = feat[torch.argsort(sums)]
    
    
    # Normalize to [0, 1]
    # feat = (feat - feat.min())/(feat.max() - feat.min())
    
    # Scaling range = [a, b] -> [0, 1]
    feat = (feat - normalize_range[0])/(normalize_range[1] - normalize_range[0])
    feat = torch.clamp(feat, min=0, max=1)

    # Convert grayscale to colormap
    if cmap is not None:
        feat = gray2cmap_tensor(feat, cmap)
        feat = torch.clamp(feat, min=0, max=1)
    else:
        feat = feat.unsqueeze(dim=1)

    feat_img = torchvision.utils.make_grid(feat, nrow=nrow, padding=2, normalize=False)
    torchvision.utils.save_image(feat_img, save_name)
    # torchvision.utils.save_image(feat, f'{save_name}')


def save_multi_tensor(img_tensor: torch.tensor, savename: str, **kwargs) -> None:
    print(img_tensor.max(), img_tensor.min())
    if img_tensor.dim() in [2, 3]:
        # (C, H, W) or (H, W)
        save_feat_grid(img_tensor, savename + '.png', **kwargs)
    elif img_tensor.dim() == 4:
    # (T, C, H, W)
        for t in range(img_tensor.shape[0]):
            save_feat_grid(img_tensor[t], savename + '_' + str(t) + '.png', **kwargs)
    else:
        print(f'tensor dim = {img_tensor.dim()} is invalid')