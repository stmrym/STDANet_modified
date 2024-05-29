import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model.edge_extractor as extractor
import models.model.blocks as blocks
from models.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
# from positional_encodings import PositionalEncodingPermute3D
from torch.nn.init import xavier_uniform_, constant_
def make_model(args):
    return ESTDAN_v1(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class ESTDAN_v1(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, sobel_out_channels=2, n_resblock=3, n_feat=32,
                 kernel_size=5, device='cuda', **kwargs):
        super(ESTDAN_v1, self).__init__()
        self.n_feat = n_feat
        InBlock = []
        InBlock.extend([nn.Sequential(
            nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1,
                        padding=3 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )])
        # print("The input of STDAN is image")
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

        self.edge_extractor = nn.Sequential(extractor.Edge_extractor_light(inplanes=1, planes=sobel_out_channels, kernel_size=3, stride=1, device=device))

        self.inBlock_channel_conv = nn.Sequential(
                        nn.Conv2d(n_feat + sobel_out_channels, n_feat, kernel_size=1, stride=1, padding='same', dilation=1),
                        nn.GELU()
        )
        self.encoder_first_channel_conv = nn.Sequential(
                        nn.Conv2d(n_feat*2 + sobel_out_channels, n_feat*2, kernel_size=1, stride=1, padding='same', dilation=1),
                        nn.GELU()
        )
        self.encoder_second_channel_conv = nn.Sequential(
                        nn.Conv2d(n_feat*4 + sobel_out_channels, n_feat*4, kernel_size=1, stride=1, padding='same', dilation=1),
                        nn.GELU()
        )

        self.orthogonal_upsampler = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding='same', dilation=1),               
                        nn.GELU(),
                        nn.PixelShuffle(4)
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
        
        # input (B*N, C, H, W) -> (B*N, 2, H, W)
        sobel_feat = self.edge_extractor(x.view(b*n, c, h, w))
        sobel_2x_downsample = F.interpolate(sobel_feat, size=(h//2, w//2),mode='bilinear', align_corners=True)
        sobel_4x_downsample = F.interpolate(sobel_feat, size=(h//4, w//4), mode='bilinear', align_corners=True)

        inblock = self.inBlock_t(x.view(b*n,c,h,w))
        # concat (B,N,32,H,W) & (B,N,2,H,W) -> (B,N,34,H,W)
        inblock = torch.cat([inblock.view(b, n, self.n_feat, h, w), sobel_feat.view(b, n, -1, h, w)], dim=2)
        # (B*N,34,H,W) -> (B*N,32,H,W)
        inblock = self.inBlock_channel_conv(inblock.view(b*n, -1, h, w))  
        
        # Encoder 1st downsampling H, W -> H/2, W/2
        encoder_1st = self.encoder_first(inblock)
        # concat (B,N,64,H/2,W/2) & (B,N,2,H/2,W/2) -> (B,N,66,H/2,W/2)
        encoder_1st = torch.cat([encoder_1st.view(b, n, self.n_feat*2, h//2, w//2), sobel_2x_downsample.view(b, n, -1, h//2, w//2)], dim=2)
        # (B,N,66,H/2,W/2) -> (B*N,64,H/2,W/2)
        encoder_1st = self.encoder_first_channel_conv(encoder_1st.view(b*n, -1, h//2, w//2))
        
        # Encoder 2nd downsampling H/2, W/2 -> H/4, W/4
        encoder_2nd = self.encoder_second(encoder_1st)
        # concat (B,N,128,H/4,W/4) & (B,N,2,H/4,W/4) -> (B,N,130,H/4,W/4)
        encoder_2nd = torch.cat([encoder_2nd.view(b, n, self.n_feat*4, h//4, w//4), sobel_4x_downsample.view(b, n, -1, h//4, w//4)], dim=2)
        # (B,N,130,H/4,W/4) -> (B,N,128,H/4,W/4)
        encoder_2nd = self.encoder_second_channel_conv(encoder_2nd.view(b*n, -1, h//4, w//4))
        
        mma_in = encoder_2nd.view(b,n,128,h//4,w//4)
        
        flow_forward,flow_backward = self.compute_flow(mma_in)
        
        frame,srcframe = self.MMA(mma_in,mma_in,flow_forward,flow_backward)
        
        mma_out = self.MSA(frame,srcframe,flow_forward,flow_backward)
        
        # (B*N, 2, H/4, W/4) -> (B, N, 2, H/4, W/4) -> center frame (B, 2, H/4, W/4)
        sobel_4x_downsample = sobel_4x_downsample.view(b,n,-1,h//4,w//4)
        # (B, 2, H/4, W/4) -> (B, 2, H/4, W/4)
        orthogonal_weight = self.orthogonal_feat_extractor(sobel_4x_downsample[:,1,:,:,:], flow_forward, flow_backward)
        orthogonal_upsample = self.orthogonal_upsampler(orthogonal_weight)

        decoder_2nd = self.decoder_second(mma_out)
        decoder_1st = self.decoder_first(decoder_2nd + encoder_1st.view(b,n,64,h//2,w//2)[:,1])
        outBlock = self.outBlock(decoder_1st + inblock.view(b,n,32,h,w)[:,1] + orthogonal_upsample)
        
        return {'out':outBlock, 'flow_forwards':flow_forward, 'flow_backwards':flow_backward, 'ortho_weight':orthogonal_weight}
