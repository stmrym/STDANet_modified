from pathlib import Path
import torch

def load_cleaning_weights(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    print(type(checkpoint))
    for k, v in checkpoint.items():
        print(k, type(v))




if __name__ == '__main__':
    ckpt_path = './exp_log/train/2024-11-15T100033_ESTDAN_v3_BSD_2ms16ms/checkpoints/ckpt-epoch-0000.pth.tar'
    load_cleaning_weights(ckpt_path)
    