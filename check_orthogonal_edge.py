import cv2
import glob
import os
import torch
from models.ESTDAN_light_Stack import ESTDAN_light_Stack

def get_weights(path, multi_file = True):
    if multi_file:
        weights = sorted(glob.glob(os.path.join(path, 'ckpt-epoch-*.pth.tar')))
    else:
        weights = [path]
    return weights


if __name__ == '__main__':

    path = './exp_log/train/F_2024-03-06T124456_ESTDAN_Stack_BSD_3ms24ms_GOPRO/checkpoints'
    weights = get_weights(path, multi_file = True)
    device = 'cpu'

    image_path = './'
    image = cv2.imread(image_path)
    output_dir = './debug_results'

    deblurnet = ESTDAN_light_Stack(device=device)

    for weight in weights[0:100]:
        checkpoint = torch.load(weight, map_location=device)
        deblurnet.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['deblurnet_state_dict'].items()})

        
        


    exit()



