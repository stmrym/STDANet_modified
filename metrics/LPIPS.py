import numpy as np
import lpips
import torch
from utils.network_utils import AverageMeter


class LPIPS(AverageMeter):
    def __init__(self, val_range=255.0):
        super(LPIPS, self).__init__()
        self.device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.val_range = val_range

    def calculate(self, img1, img2, **kwargs):
        img1 = self._convert_to_tensor(img1, self.val_range) 
        img2 = self._convert_to_tensor(img2, self.val_range) 
        loss = self.loss_fn_alex(img1.permute(2,0,1).unsqueeze(0), img2.permute(2,0,1).unsqueeze(0)).mean().detach().cpu()
        return loss.cpu().item()

    def _convert_to_tensor(self, img, val_range):
        if not isinstance(img, torch.Tensor):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.astype(np.float32) / val_range).to(self.device)
            else:
                print('unsupported format')
                exit()
        return img