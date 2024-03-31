import torch
from models.ESTDAN_Stack import ESTDAN_Stack

weights = './exp_log/train/F_2024-03-06T124456_ESTDAN_Stack_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-1200.pth.tar'

checkpoint = torch.load(weights,map_location='cpu')
# print(type(checkpoint))
deblurnet = checkpoint['deblurnet_state_dict']

# sobel_weights = deblurnet['module.recons_net.edge_extractor.0.']

for k, v in deblurnet.items():
    if 'module.recons_net.edge_extractor' in k: 
        print(k)
        print(v)

# print(deblurnet.keys())

