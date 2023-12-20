import torch
import torch.nn as nn
from models.model import STDAN_RAFT
# from models.model import flow_pwc
from utils import util
import torch.nn.functional as F

class STDAN_RAFT_Stack(nn.Module):

    def __init__(self, in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
                    load_flow_net=True, load_recons_net=False, use_raft_flow=False, flow_pretrain_fn='/home/hczhang/CODE/CDVD-TSP/pretrain_models/network-default.pytorch', recons_pretrain_fn='',
                    device='cuda'):
        super(STDAN_RAFT_Stack, self).__init__()

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 5, "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.recons_net = STDAN_RAFT.STDAN_RAFT(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat, use_raft_flow=use_raft_flow)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def down_size(self,frame):
        _,_,h,w = frame.shape
        frame = F.interpolate(frame, size=(h//4, w//4),mode='bilinear', align_corners=True)
        return frame
    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        concated = torch.stack([frame_list[0], frame_list[1], frame_list[2]], dim=1)

        recons_1,flow_forward_recons_1,flow_backward_recons_1 = self.recons_net(concated, use_raft_flow=False)

        
        concated = torch.stack([frame_list[1], frame_list[2], frame_list[3]], dim=1)
        recons_2, flow_forward_recons_2,flow_backward_recons_2 = self.recons_net(concated, use_raft_flow=False)

        
        concated = torch.stack([frame_list[2], frame_list[3], frame_list[4]], dim=1)
        recons_3, flow_forward_recons_3,flow_backward_recons_3 = self.recons_net(concated, use_raft_flow=False)

        
        concated = torch.stack([recons_1, recons_2, recons_3], dim=1) 

        out, flow_forward_out,flow_backward_out = self.recons_net(concated, use_raft_flow=True)
        
        flow_forwards = [flow_forward_recons_1,flow_forward_recons_2,flow_forward_recons_3,flow_forward_out]
        flow_backwards = [flow_backward_recons_1,flow_backward_recons_2,flow_backward_recons_3,flow_backward_out]
        return recons_1, recons_2, recons_3, out,flow_forwards,flow_backwards
