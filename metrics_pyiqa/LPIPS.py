import pyiqa
import sys
import os

sys.path.append(os.path.dirname(__file__))

from LR_utils.tensor_util import filepathlist2tensor
from LR_utils.stop_watch import stop_watch


class LPIPS:
    def __init__(self, device):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)

    # @stop_watch
    def calculate(self, recons, gt, **kwargs):
        result = self.iqa_metric(recons, gt)
        return result.cpu().item()


if __name__ == '__main__':

    device = 'cuda:0'
    
    recons_l = [
        '/mnt/d/results/20241210/074_00000034_output.png'
    ]

    gt_l = [
        '/mnt/d/results/20241210/074_00000034_gt.png'
    ]

    recons_tensor = filepathlist2tensor(recons_l, device)
    gt_tensor = filepathlist2tensor(gt_l, device)

    metric = LPIPS(device)

    result = metric.calculate(recons=recons_tensor, gt=gt_tensor)
    print(result)
    