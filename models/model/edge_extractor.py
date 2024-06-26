import torch
import torch.nn as nn
import torch.nn.init as init


###############################
# Edge Extractor
###############################

class Edge_extractor(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(Edge_extractor, self).__init__()
        a = torch.tensor(1.0)
        sobel_kernel = torch.cuda.FloatTensor(
            [[  [  -a, 0,   a],
                [-2*a, 0, 2*a],
                [  -a, 0,   a]],

            [   [-a, -2*a, -a],
                [ 0,    0,  0],
                [ a,  2*a,  a]],
                
            [   [-2*a, -a,   0],
                [  -a,  0,   a],
                [   0,  a, 2*a]],
                
            [   [   0,  a, 2*a],
                [  -a,  0,   a],
                [-2*a, -a,   0]]
                ])
        # (4, 1, 3, 3)
        sobel_kernel = nn.Parameter(sobel_kernel.unsqueeze(dim=1)) 

        self.sobel_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                            padding='same', dilation=dilation, padding_mode='reflect')
        self.sobel_conv.weight = sobel_kernel
        self.gelu = nn.GELU()

    def forward(self, x):     
        bn, c, h, w = x.shape
        assert (c == 3 or c == 1), f'Input channel {c} invalid!'       
        if c == 3:
            grayscale_x = 0.114*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.299*x[:,2,:,:]
        elif c == 1:
            pass
        
        # (BN, H, W) -> (BN, 1, H, W)
        grayscale_x = grayscale_x.unsqueeze(dim=1)
        # Input (BN, 1, H, W) -> Output (BN, 4, H, W)
        sobel_out = self.gelu(self.sobel_conv(grayscale_x))
        return sobel_out
    


class Edge_extractor_light(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, device='cuda'):
        super(Edge_extractor_light, self).__init__()
        a = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        sobel_kernel = torch.tensor(
            [[  [  -a, 0,   a],
                [-2*a, 0, 2*a],
                [  -a, 0,   a]],

            [   [-a, -2*a, -a],
                [ 0,    0,  0],
                [ a,  2*a,  a]]
                ], dtype=torch.float32, device=device)

        self.sobel_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                            padding='same', dilation=dilation, padding_mode='reflect')
        # (2, 1, 3, 3)
        self.sobel_conv.weight = nn.Parameter(sobel_kernel.unsqueeze(dim=1))
        # initialize bias to 0
        init.constant_(self.sobel_conv.bias, 0.0)
        self.gelu = nn.GELU()

    def forward(self, x):     
        bn, c, h, w = x.shape
        assert (c == 3 or c == 1), f'Input channel {c} invalid!'       
        if c == 3:
            grayscale_x = 0.114*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.299*x[:,2,:,:]
        elif c == 1:
            pass
        
        # (BN, H, W) -> (BN, 1, H, W)
        grayscale_x = grayscale_x.unsqueeze(dim=1)
        # Input (BN, 1, H, W) -> Output (BN, 2, H, W)
        sobel_out = self.gelu(self.sobel_conv(grayscale_x))
        return sobel_out



