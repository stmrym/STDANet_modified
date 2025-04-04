import torch.nn as nn
import torch
import models.model.blocks as blocks
from models.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
from models.mmedit import ResidualBlocksWithInputConv
# from positional_encodings import PositionalEncodingPermute3D
from torch.nn.init import xavier_uniform_, constant_
def make_model(args):
    return STDAN(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class STDAN(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_resblock=3, n_feat=32, kernel_size=5, 
                 use_cleaning=False, is_sequential_cleaning=False, is_fix_cleaning=False, dynamic_refine_thres=255, n_cleaning_blocks=20, mid_channels=64,
                 **kwargs):
        super(STDAN, self).__init__()

        self.n_feat = n_feat
        self.use_cleaning = use_cleaning
        
        if self.use_cleaning:
            self.dynamic_refine_thres = dynamic_refine_thres / 255.
            self.is_sequential_cleaning = is_sequential_cleaning

            # image cleaning module
            self.image_cleaning = nn.Sequential(
                ResidualBlocksWithInputConv(3, mid_channels, n_cleaning_blocks),
                nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
            )
            if is_fix_cleaning:  # keep the weights of the cleaning module fixed
                self.image_cleaning.requires_grad_(False)
            print('Use cleaning module')

        InBlock = []
        InBlock.extend([nn.Sequential(
            nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1,
                        padding=3 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=3, stride=1)
                        for _ in range(3)])
        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=3, stride=1)
                              for _ in range(3)])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=3, stride=1)
                               for _ in range(3)])

        # decoder2
        Decoder_second = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)]
        Decoder_second.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        ))
        # decoder1
        Decoder_first = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                         for _ in range(n_resblock)]
        Decoder_first.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]
        OutBlock.append(
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

        self.inBlock_t = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)

        

        self.MMA = DeformableAttnBlock(n_heads=4,d_model=128,n_levels=3,n_points=12)
        # self.Defattn2 = DeformableAttnBlock(n_heads=8,d_model=128,n_levels=3,n_points=12)
        self.MSA = DeformableAttnBlock_FUSION(n_heads=4,d_model=128,n_levels=3,n_points=12)
        
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=2*n_feat * 4, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(0.1,inplace=True),
        )
        self.motion_out = torch.nn.Conv2d(in_channels=32//2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        constant_(self.motion_out.weight.data, 0.)
        constant_(self.motion_out.bias.data, 0.)

    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.estimate_flow(frames_1, frames_2).view(n, t-1, 2, h, w)

        flows_backward = self.estimate_flow(frames_2,frames_1).view(n, t-1, 2, h, w)

        return flows_forward,flows_backward
    
    def estimate_flow(self,frames_1, frames_2):
        return self.motion_out(self.motion_branch(torch.cat([frames_1, frames_2],1)))
        
    def forward(self, x):
        b, n, c, h, w = x.size()
        
                # Pre-Cleaning Module
        if self.use_cleaning:
            for clean_iter in range(0, 3):  # at most 3 cleaning, determined empirically
                if self.is_sequential_cleaning:
                    residues = []
                    for i in range(0, n):
                        residue_i = self.image_cleaning(x[:, i, :, :, :])
                        x[:, i, :, :, :] += residue_i
                        residues.append(residue_i)
                    residues = torch.stack(residues, dim=1)
                else:  # time -> batch, then apply cleaning at once
                    x = x.view(-1, c, h, w)
                    residues = self.image_cleaning(x)
                    x = (x + residues).view(b, n, c, h, w)
                
                # determine whether to continue cleaning
                # print(clean_iter, torch.mean(torch.abs(residues)))
                if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                    break
        
        first_scale_inblock = self.inBlock_t(x.view(b*n,c,h,w))
        
        first_scale_encoder_first = self.encoder_first(first_scale_inblock)
        
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
        first_scale_encoder_second = first_scale_encoder_second.view(b,n,128,h//4,w//4)
        
        flow_forward,flow_backward = self.compute_flow(first_scale_encoder_second)
        
        frame,srcframe = self.MMA(first_scale_encoder_second,first_scale_encoder_second,flow_forward,flow_backward)
        
        first_scale_encoder_second = self.MSA(frame,srcframe,flow_forward,flow_backward)
        
        first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first.view(b,n,64,h//2,w//2)[:,1])
        
        
        first_scale_outBlock = self.outBlock(first_scale_decoder_first+first_scale_inblock.view(b,n,32,h,w)[:,1])
        
        return {'pre_input' : x,
                'out' : first_scale_outBlock, 
                'flow_forwards' : flow_forward, 
                'flow_backwards' : flow_backward}
