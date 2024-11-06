import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

class Stack(nn.Module):

    def __init__(self, n_sequence, arch, device, use_stack, **kwargs):
        super(Stack, self).__init__()

        self.n_sequence = n_sequence
        self.device = device
        self.arch = arch
        self.use_stack = use_stack

        self._module = importlib.import_module('models.model.' + self.arch)
        self.recons_net = self._module.__dict__[self.arch](device=device, **kwargs)

    def down_size(self,frame):
        _,_,h,w = frame.shape
        frame = F.interpolate(frame, size=(h//4, w//4),mode='bilinear', align_corners=True)
        return frame
    
    def forward(self, x):

        if self.use_stack:
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