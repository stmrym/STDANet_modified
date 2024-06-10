import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model.edge_extractor as extractor
import models.model.blocks as blocks
from models.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
# from positional_encodings import PositionalEncodingPermute3D
from torch.nn.init import xavier_uniform_, constant_
def make_model(args):
    return ESTDAN_v2_debug(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class ESTDAN_v2_debug(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, sobel_out_channels=2, n_resblock=3, n_feat=32,
                 kernel_size=5, device='cuda', **kwargs):
        super(ESTDAN_v2_debug, self).__init__()
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
            nn.Conv2d(n_feat + sobel_out_channels, n_feat*2, kernel_size=3, stride=2, padding=3//2),
            nn.LeakyReLU(0.1,inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat*2, n_feat*2, kernel_size=3, stride=1)
                            for _ in range(3)])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat*2 + sobel_out_channels, n_feat*4 - 3, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat*4 - 3, n_feat*4 - 3, kernel_size=3, stride=1)
                            for _ in range(3)])


        # decoder2
        Decoder_second = [blocks.ResBlock(n_feat*4 + 2, n_feat*4 + 2, kernel_size=kernel_size, stride=1)
                            for _ in range(n_resblock)]
        Decoder_second.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat*4 + 2, n_feat*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        ))
        # decoder1
        Decoder_first = [blocks.ResBlock(n_feat*2 + 2, n_feat*2 + 2, kernel_size=kernel_size, stride=1)
                            for _ in range(n_resblock)]
        Decoder_first.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat*2 + 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat + 2, n_feat + 2, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]
        OutBlock.append(
            nn.Conv2d(n_feat + 2, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

        self.inBlock_t = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)

        self.edge_extractor = nn.Sequential(extractor.Edge_extractor_light(inplanes=1, planes=sobel_out_channels, kernel_size=3, stride=1, device=device))

        # self.inBlock_channel_conv = nn.Sequential(
        #                 nn.Conv2d(n_feat + sobel_out_channels, n_feat, kernel_size=1, stride=1, padding='same', dilation=1),
        #                 nn.GELU()
        # )
        # self.encoder_first_channel_conv = nn.Sequential(
        #                 nn.Conv2d(n_feat*2 + sobel_out_channels, n_feat*2, kernel_size=1, stride=1, padding='same', dilation=1),
        #                 nn.GELU()
        # )
        # self.encoder_second_channel_conv = nn.Sequential(
        #                 nn.Conv2d(n_feat*4 + sobel_out_channels, n_feat*4, kernel_size=1, stride=1, padding='same', dilation=1),
        #                 nn.GELU()
        # )

        self.orthogonal_second_upsampler = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding='same', dilation=1),
                        nn.GELU(),
                        nn.PixelShuffle(2)
        )
        self.orthogonal_first_upsampler = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding='same', dilation=1),               
                        nn.GELU(),
                        nn.PixelShuffle(2)
        )

        self.MMA = DeformableAttnBlock(n_heads=4, d_model=128, n_levels=3, n_points=12)
        # self.Defattn2 = DeformableAttnBlock(n_heads=8,d_model=128,n_levels=3,n_points=12)
        self.MSA = DeformableAttnBlock_FUSION(n_heads=4, d_model=128, n_levels=3, n_points=12)
        
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = nn.Sequential(
                    nn.Conv2d(in_channels=2*(n_feat*4 - 3), out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
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

        # flow_forward[:,0] : frame1 -> frame2
        # flow_forward[:,1] : frame2 -> frame3
        # flow_backward[:,0] : frame2 -> frame1
        # flow_backward[:,1] : frame3 -> frame2

        frame1_weight = torch.mul(flow_forward[:,0,0,:,:], edge[:,0,0,:,:]) + torch.mul(flow_forward[:,0,1,:,:], edge[:,0,1,:,:])
        frame2_f_weight = torch.mul(flow_forward[:,1,0,:,:], edge[:,1,0,:,:]) + torch.mul(flow_forward[:,1,1,:,:], edge[:,1,1,:,:])
        frame2_b_weight = torch.mul(flow_backward[:,0,0,:,:], edge[:,1,0,:,:]) + torch.mul(flow_backward[:,0,1,:,:], edge[:,1,1,:,:])
        frame3_weight = torch.mul(flow_backward[:,1,0,:,:], edge[:,2,0,:,:]) + torch.mul(flow_backward[:,1,1,:,:], edge[:,2,1,:,:])

        orthogonal_weights = (torch.stack([frame1_weight, frame2_f_weight, frame2_b_weight, frame3_weight], dim=1)).unsqueeze(dim=2)
        # (B, 4, 1, H/4, W/4)
        return torch.abs(orthogonal_weights)
        # return orthogonal_weights
    
    def forward(self, x):
        b, n, c, h, w = x.size()
        # input (B*N, C, H, W) -> (B*N, 2, H, W)
        sobel_feat = self.edge_extractor(x.view(b*n, c, h, w))
        sobel_2x_downsample = F.interpolate(sobel_feat, size=(h//2, w//2),mode='bilinear', align_corners=True)
        sobel_4x_downsample = F.interpolate(sobel_feat, size=(h//4, w//4), mode='bilinear', align_corners=True)

        inblock = self.inBlock_t(x.view(b*n,c,h,w))
        # concat (B*N,32,H,W) & (B*N,2,H,W) -> (B*N,34,H,W)
        inblock = torch.cat([inblock, sobel_feat], dim=1)
        # (B*N,34,H,W) -> (B*N,32,H,W)
        # inblock = self.inBlock_channel_conv(inblock.view(b*n, -1, h, w))  
        
        # Encoder 1st downsampling H, W -> H/2, W/2
        encoder_1st = self.encoder_first(inblock)
        # concat (B*N,64,H/2,W/2) & (B*N,2,H/2,W/2) -> (B*N,66,H/2,W/2)
        encoder_1st = torch.cat([encoder_1st, sobel_2x_downsample], dim=1)
        # (B,N,66,H/2,W/2) -> (B*N,64,H/2,W/2)
        # encoder_1st = self.encoder_first_channel_conv(encoder_1st.view(b*n, -1, h//2, w//2))
        
        # Encoder 2nd downsampling H/2, W/2 -> H/4, W/4
        encoder_2nd = self.encoder_second(encoder_1st)
        # concat (B*N,128,H/4,W/4) & (B*N,2,H/4,W/4) -> (B*N,130,H/4,W/4)
        # encoder_2nd = torch.cat([encoder_2nd, sobel_4x_downsample], dim=1)
        # (B,N,130,H/4,W/4) -> (B,N,128,H/4,W/4)
        # encoder_2nd = self.encoder_second_channel_conv(encoder_2nd.view(b*n, -1, h//4, w//4))
        
        # Estimate flow
        flow_forward,flow_backward = self.compute_flow(encoder_2nd.view(b,n,-1,h//4,w//4))
        
        # (B*N, 2, H/4, W/4) -> (B, N+1, 1, H/4, W/4)
        orthogonal_weights = self.orthogonal_feat_extractor((sobel_4x_downsample.view(b,n,-1,h//4,w//4)), flow_forward, flow_backward)
        
        # (B, N, 1, H/4, W/4) : (frame1, (frame2_f + frame2_b)/2, frame3)
        orthogonal_mma_in_weights = (torch.stack([orthogonal_weights[:,0], (orthogonal_weights[:,1]+orthogonal_weights[:,2])/2, orthogonal_weights[:,3]], dim=1))
        orthogonal_mma_in_weights = orthogonal_mma_in_weights.view(b*n,1,h//4,w//4)

        # concat (B*N,125,H/4,W/4) & (B*N,2,H/4,W/4) & (B*N,1,H/4,W/4) -> (B,N,128,H/4,W/4)
        mma_in = (torch.cat([encoder_2nd, sobel_4x_downsample, orthogonal_mma_in_weights], dim=1)).view(b,n,-1,h//4,w//4)
        frame,srcframe = self.MMA(mma_in,mma_in,flow_forward,flow_backward)
        # (B, 128, H/4, W/4)
        mma_out = self.MSA(frame,srcframe,flow_forward,flow_backward)
        
        # (B, 2, 1, H/4, W/4) -> (B, 2, H/4, W/4)
        orthogonal_center_weights = torch.squeeze(orthogonal_weights[:,1:3,:,:,:], dim=2)
        orthogonal_2nd_upsample = self.orthogonal_second_upsampler(orthogonal_center_weights)
        orthogonal_1st_upsample = self.orthogonal_first_upsampler(orthogonal_2nd_upsample)

        # (B, 130, H/4, W/4) -> (B, 64, H/2, W/2)
        mma_out = torch.cat([mma_out, orthogonal_center_weights], dim=1)
        decoder_2nd = self.decoder_second(mma_out)
        decoder_2nd = torch.cat([decoder_2nd, orthogonal_2nd_upsample], dim=1)
        
        # (B, 66, H/2, W/2) -> (B, 32, H, W)
        decoder_1st = self.decoder_first(decoder_2nd + encoder_1st.view(b,n,-1,h//2,w//2)[:,1])
        decoder_1st = torch.cat([decoder_1st, orthogonal_1st_upsample], dim=1)

        # (B, 34, H, W) -> (B, 3, H, W)
        outBlock = self.outBlock(decoder_1st + inblock.view(b,n,-1,h,w)[:,1])

        # sobel_4x_downsample = torch.sqrt((sobel_4x_downsample[:,0]**2 + sobel_4x_downsample[:,1]**2))
        # return {'out':outBlock, 'flow_forwards':flow_forward, 'flow_backwards':flow_backward, 'ortho_weight':orthogonal_center_weights}
        return {'out':outBlock, 'flow_forwards':flow_forward, 'flow_backwards':flow_backward, 
            'first_scale_inblock': inblock.view(b,n,-1,h,w), 'first_scale_encoder_first':encoder_1st.view(b,n,-1,h//2,w//2),
            'first_scale_encoder_second':mma_in, 'first_scale_encoder_second_out':mma_out,
            'first_scale_decoder_second':decoder_2nd, 'first_scale_decoder_first':decoder_1st,
            'sobel_edge':sobel_4x_downsample.view(b,n,-1,h//4,w//4), 'motion_orthogonal_edge':orthogonal_center_weights
            }
    
