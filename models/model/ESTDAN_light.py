import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model.edge_extractor as extractor
import models.model.blocks as blocks
from models.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
# from positional_encodings import PositionalEncodingPermute3D
from torch.nn.init import xavier_uniform_, constant_
def make_model(args):
    return ESTDAN_light(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class ESTDAN_light(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, sobel_out_channels=2, n_resblock=3, n_feat=32,
                 kernel_size=5, feat_in=False, n_in_feat=32):
        super(ESTDAN_light, self).__init__()

        self.feat_in = feat_in

        InBlock = []
        if not feat_in:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1,
                          padding=3 // 2),
                nn.LeakyReLU(0.1,inplace=True)
            )])
            # print("The input of STDAN is image")
        else:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(n_in_feat, n_feat, kernel_size=3, stride=1, padding=3 // 2),
                nn.LeakyReLU(0.1,inplace=True)
            )])
            # print("The input of STDAN is feature")
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

        self.edge_extractor = nn.Sequential(extractor.Edge_extractor_light(inplanes=1, planes=sobel_out_channels, kernel_size=3, stride=1))

        self.orthogonal_feat_conv = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=n_feat*4, kernel_size=3, stride=1, padding='same', dilation=1),
                        nn.GELU(),
        )

        self.orthogonal_second_upsampler = nn.Sequential(
                        nn.Conv2d(in_channels=n_feat*4, out_channels=n_feat*8, kernel_size=3, stride=1, padding='same', dilation=1),
                        nn.GELU(),
                        nn.PixelShuffle(2)
        )
        self.orthogonal_first_upsampler = nn.Sequential(
                        nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat*4, kernel_size=3, stride=1, padding='same', dilation=1),               
                        nn.GELU(),
                        nn.PixelShuffle(2)
        )

        self.MMA = DeformableAttnBlock(n_heads=4,d_model=128,n_levels=3,n_points=12)
        # self.Defattn2 = DeformableAttnBlock(n_heads=8,d_model=128,n_levels=3,n_points=12)
        self.MSA = DeformableAttnBlock_FUSION(n_heads=4,d_model=128,n_levels=3,n_points=12)
        
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = nn.Sequential(
                    nn.Conv2d(in_channels=2*n_feat * 4, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(0.1,inplace=True),
                    nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(0.1,inplace=True),
                    nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(0.1,inplace=True),
        )
        self.motion_out = nn.Conv2d(in_channels=32//2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
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
        
    def orthogonal_feat_extractor(self, edge, flow_forward, flow_backward):
        out_flow_f = flow_forward[:,1,:,:,:]
        out_flow_b = flow_backward[:,0,:,:,:]

        orthogonal_forward_weight = torch.mul(out_flow_f[:,0,:,:], edge[:,0,:,:]) + torch.mul(out_flow_f[:,1,:,:], edge[:,1,:,:])
        orthogonal_backward_weight = torch.mul(out_flow_b[:,0,:,:], edge[:,0,:,:]) + torch.mul(out_flow_b[:,1,:,:], edge[:,1,:,:])

        orthogonal_weight = torch.stack([torch.abs(orthogonal_forward_weight), torch.abs(orthogonal_backward_weight)], dim=1)
        return orthogonal_weight

    def forward(self, x):
        b, n, c, h, w = x.size()
        
        first_scale_inblock = self.inBlock_t(x.view(b*n,c,h,w))
        first_scale_encoder_first = self.encoder_first(first_scale_inblock)
        
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
        first_scale_encoder_second = first_scale_encoder_second.view(b,n,128,h//4,w//4)
        
        flow_forward,flow_backward = self.compute_flow(first_scale_encoder_second)
        
        frame,srcframe = self.MMA(first_scale_encoder_second,first_scale_encoder_second,flow_forward,flow_backward)
        
        first_scale_encoder_second = self.MSA(frame,srcframe,flow_forward,flow_backward)
        
        # input (B, C, H, W) -> (B, 2, H, W)
        sobel_feat = self.edge_extractor(x[:,1,:,:,:])
        sobel_feat_downsampled = F.interpolate(sobel_feat, size=(h//4, w//4),mode='bilinear', align_corners=True)

        # (B, 2, 2, H, W) -> (B, 2, H, W)
        orthogonal_weight = self.orthogonal_feat_extractor(sobel_feat_downsampled, flow_forward, flow_backward)
        orthogonal_feat = self.orthogonal_feat_conv(orthogonal_weight)
        orthogonal_feat_second = self.orthogonal_second_upsampler(orthogonal_feat)
        orthogonal_feat_first = self.orthogonal_first_upsampler(orthogonal_feat_second)

        first_scale_decoder_second = self.decoder_second(first_scale_encoder_second + orthogonal_feat)
        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first.view(b,n,64,h//2,w//2)[:,1] + orthogonal_feat_second)
        first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock.view(b,n,32,h,w)[:,1] + orthogonal_feat_first)
        
        return first_scale_outBlock, flow_forward,flow_backward
