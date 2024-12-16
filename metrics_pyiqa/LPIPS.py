import pyiqa
import sys
import os

sys.path.append(os.path.dirname(__file__))

from LR_utils.stop_watch import stop_watch

class LPIPS:
    def __init__(self, device):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)

    # @stop_watch
    def calculate(self, recons, gt, **kwargs):
        result = self.iqa_metric(recons, gt)
        return result.cpu().item()


    