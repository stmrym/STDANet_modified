import importlib
import torch
import torch.nn as nn
from models.model import ESTDAN_light
import torch.nn.functional as F

class Stack(nn.Module):

    def __init__(self, network_arch, use_stack = True, in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32, device='cuda'):
        super(Stack, self).__init__()

        self.n_sequence = n_sequence
        self.device = device
        self._network_arch = network_arch
        self._use_stack = use_stack

        self._module = importlib.import_module('models.model.' + self._network_arch)
        self.recons_net = self._module.__dict__[self._network_arch](in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat, device=device)

    def down_size(self,frame):
        _,_,h,w = frame.shape
        frame = F.interpolate(frame, size=(h//4, w//4),mode='bilinear', align_corners=True)
        return frame
    
    def forward(self, x):

        if self._use_stack:
            assert self.n_sequence == 5, 'Please set DATA.INPUT_LENGTH to 5.'

            frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
            concated = torch.stack([frame_list[0], frame_list[1], frame_list[2]], dim=1)

            recons_1 = self.recons_net(concated)

            concated = torch.stack([frame_list[1], frame_list[2], frame_list[3]], dim=1)
            recons_2 = self.recons_net(concated)

            concated = torch.stack([frame_list[2], frame_list[3], frame_list[4]], dim=1)
            recons_3 = self.recons_net(concated)
            
            concated = torch.stack([recons_1['out'], recons_2['out'], recons_3['out']], dim=1) 

            final = self.recons_net(concated)
            
            output_dict = {}
            for key in recons_1.keys():
                output_dict[key] = {}
                output_dict[key]['recons_1'] = recons_1[key]
                output_dict[key]['recons_2'] = recons_2[key]
                output_dict[key]['recons_3'] = recons_3[key]
                output_dict[key]['final'] = final[key]

        else:
            assert self.n_sequence == 3, 'Please set DATA.INPUT_LENGTH to 3.'

            final = self.recons_net(x)
            output_dict = {}
            for key in final.keys():
                output_dict[key] = {}
                output_dict[key]['final'] = final[key]            
        
        return output_dict