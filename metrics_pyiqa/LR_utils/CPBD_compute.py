import kornia
import numpy as np
import torch
import torch.nn.functional as F
from math import pi
import sys
import os 

sys.path.append(os.path.dirname(__file__))

from tensor_util import tensor_rgb2gray
from grad_util import gradient_cuda
from stop_watch import stop_watch

# threshold to characterize blocks as edge/non-edge blocks
THRESHOLD = 0.002

# fitting parameter
BETA = 3.6

# block size
BLOCK_HEIGHT, BLOCK_WIDTH = (64, 64)

# just noticeable widths based on the perceptual experiments
WIDTH_JNB = np.concatenate([5*np.ones(51), 3*np.ones(205)])



def _check_input_tensor(image):
    '''
    image: torch.tensor (H, W) or (C, H, W)

    ->

    image: torch.tensor (B, 1, H, W)
    '''
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(1)
    elif len(image.shape) == 4:
        assert image.shape[1] == 3, f'Unsupported input channel {image.shape[1]}'
        image = tensor_rgb2gray(image).unsqueeze(1)
    
    assert len(image.shape) >= 2 and len(image.shape) <= 4, f'Unsupported input dims {image.shape}'

    return image
        

def _sobel(image):
    '''
    image: torch.tensor (B, 1, H, W)

    ->

    torch.tensor (H, W)
    '''
    HSOBEL_WEIGHTS = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    h1 = torch.tensor(HSOBEL_WEIGHTS, dtype=torch.float32, device=image.device)
    h1 /= torch.sum(torch.abs(h1))  # normalize h1

    image = F.pad(image, (1,1,1,1), mode='reflect')
    strength2 = torch.square(F.conv2d(image, h1.T.unsqueeze(0).unsqueeze(0)))
 
    thresh2 = 2 * torch.sqrt(torch.mean(strength2))

    strength2[strength2 <= thresh2] = 0
    return _simple_thinning(strength2.squeeze())


def _simple_thinning(strength):
    """
    Perform a very simple thinning.
    """
    num_rows, num_cols = strength.shape

    zero_column = torch.zeros((num_rows, 1), device=strength.device)
    zero_row = torch.zeros((1, num_cols), device=strength.device)

    x = (
        (strength > torch.cat([zero_column, strength[:, :-1]], dim=1)) &
        (strength > torch.cat([strength[:, 1:], zero_column], dim=1))
    )

    y = (
        (strength > torch.cat([zero_row, strength[:-1, :]], dim=0)) &
        (strength > torch.cat([strength[1:, :], zero_row], dim=0))
    )

    return x | y


def _initialize_cond(h, w, device):
    cond = torch.zeros((h, w), device=device, dtype=torch.bool)
    cond[1:-1, 1:-1] = True
    return cond


def cpbd_compute(image):
    '''
    image: torch.Tensor (Gray) [0,1] with shape (B, H, W) or (B, 1, H, W)
    '''
    assert isinstance(image, torch.Tensor)
    image = _check_input_tensor(image)
    # image = image*255.0
    image = image.float()

    # (B, 1, H, W) -> (B, 1, H, W)
    _, canny_edges = kornia.filters.canny(image, low_threshold=0.1, high_threshold=0.2)
    sobel_edges = _sobel(image)

    marziliano_widths = _marziliano_method(sobel_edges, image[0,0])


    sharpness_metric = _calculate_sharpness_metric(image[0,0], canny_edges[0,0], marziliano_widths)


    # from cpbd.compute import _calculate_sharpness_metric as _cs
    # from cpbd.compute import marziliano_method as mm
    # import numpy as np
    # from skimage.feature import canny
    # from cpbd.octave import sobel

    # image2 = image[0,0].cpu().numpy()

    # canny_edges2 = canny(image2)
    # sobel_edges2 = sobel(image2)

    # # edge width calculation
    # marziliano_widths2 = mm(sobel_edges2, image2)
    # res =  _cs(image2, canny_edges2, marziliano_widths2)

    # print(res)


    # exit()

    return sharpness_metric

# @stop_watch
def _marziliano_method(edges, image):
    '''
    edges: torch.tensor [0,1] with shape (H, W)
    image: torch.tensor [0,1] with shape (H, W)
    '''
    """
    Calculate the widths of the given edges.

    :return: A tensor with the same dimensions as the given image with 0's at
        non-edge locations and edge-widths at the edge locations.
    """

    # `edge_widths` consists of zero and non-zero values. A zero value
    # indicates that there is no edge at that position and a non-zero value
    # indicates that there is an edge at that position and the value itself
    # gives the edge width.
    edge_widths = torch.zeros_like(image, device=image.device)

    # find the gradient for the image
    # gradient_y, gradient_x = torch.gradient(image)
    gradient_y, gradient_x = gradient_cuda(image)


    # dimensions of the image
    h, w = image.shape
    # holds the angle information of the edges
    edge_angles = torch.zeros_like(image, device=image.device)

    # calculate the angle of the edges
    mask = (gradient_x != 0) 
    edge_angles[mask] = torch.atan2(gradient_y[mask], gradient_x[mask]) * (180 / pi) 
    
    # Set angles to 0 where both gradient_x and gradient_y are zero 
    mask_zero = (gradient_x == 0) & (gradient_y == 0) 
    edge_angles[mask_zero] = 0 
    
    # Set angles to 90 where gradient_x is zero and gradient_y is pi/2 
    mask_ninety = (gradient_x == 0) & (gradient_y == pi / 2) 
    edge_angles[mask_ninety] = 90


    # if torch.any(edge_angles):

    #     # Quantize the angle
    #     quantized_angles = 45 * torch.round(edge_angles / 45)

    #     for row in range(1, h - 1):
    #         for col in range(1, w - 1):
    #             if edges[row, col] == 1:
    #                 # print(row, col, quantized_angles[row, col])
    #                 # Gradient angle = 180 or -180
    #                 if quantized_angles[row, col] == 180 or quantized_angles[row, col] == -180:
    #                     for margin in range(101):
    #                         inner_border = (col - 1) - margin
    #                         outer_border = (col - 2) - margin

    #                         if row==503 and col==486:
    #                             print(image[row, outer_border] - image[row, inner_border], margin+1)

    #                         if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) <= 0:
    #                             # print(f'180l----{margin}')
    #                             break

    #                     width_left = margin + 1
    #                     # print(row, col, width_left)

    #                     for margin in range(101):
    #                         inner_border = (col + 1) + margin
    #                         outer_border = (col + 2) + margin

    #                         if outer_border >= w or (image[row, outer_border] - image[row, inner_border]) >= 0:
    #                             # print(f'180r----{margin}')

    #                             break

    #                     width_right = margin + 1

    #                     edge_widths[row, col] = width_left + width_right

    #                 # Gradient angle = 0
    #                 if quantized_angles[row, col] == 0:
    #                     for margin in range(101):
    #                         inner_border = (col - 1) - margin
    #                         outer_border = (col - 2) - margin




    #                         if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) >= 0:
    #                             break

    #                     width_left = margin + 1

    #                     for margin in range(101):
    #                         inner_border = (col + 1) + margin
    #                         outer_border = (col + 2) + margin

    #                         if outer_border >= w or (image[row, outer_border] - image[row, inner_border]) <= 0:
    #                             # print(f'0r----{margin}')
    #                             break

    #                     width_right = margin + 1

    #                     edge_widths[row, col] = width_right + width_left

    #     edge_widths2 = edge_widths.clone()

    #     # print(edge_widths2)
    #     # count_values(edge_widths2)
    #     # print(edge_widths2.max(), edge_widths2.min())
    #     # exit()
    if torch.any(edge_angles):

        quantized_angles = 45 * torch.round(edge_angles / 45)

        # Masks for edges and angles
        mask_edge = edges == 1

        margin_range = torch.arange(101, device=image.device, dtype=torch.int8)

        # Calculate widths for gradient angle = 180 or -180
        mask_angle_180 = mask_edge & ((quantized_angles == 180) | (quantized_angles == -180))
        mask_angle_0 = mask_edge & (quantized_angles == 0)

        left_widths_180 = torch.zeros((h, w), device=image.device, dtype=torch.int8)
        right_widths_180 = torch.zeros((h, w), device=image.device, dtype=torch.int8)
        left_widths_0 = torch.zeros((h, w), device=image.device, dtype=torch.int8)
        right_widths_0 = torch.zeros((h, w), device=image.device, dtype=torch.int8)

        left_cond_180 = _initialize_cond(h, w, image.device)
        right_cond_180 = _initialize_cond(h, w, image.device)
        left_cond_0 = _initialize_cond(h, w, image.device)
        right_cond_0 = _initialize_cond(h, w, image.device)

        for margin in margin_range:

            # decide left side
            inner_border_left = torch.clamp(torch.arange(w, device=image.device) - (1 + margin), min=0).long()
            outer_border_left = torch.clamp(torch.arange(w, device=image.device) - (2 + margin), min=0).long()

            left_diff = image[:, outer_border_left] - image[:, inner_border_left]
            
            left_widths_180 = torch.where(left_cond_180, margin + 1, left_widths_180)            
            left_cond_180 = (outer_border_left >= 0) & (left_diff > 0) & left_cond_180

            left_widths_0 = torch.where(left_cond_0, margin + 1, left_widths_0)
            left_cond_0 = (outer_border_left >= 0) & (left_diff < 0) & left_cond_0

        
            # decide right side        
            inner_border_right = torch.clamp(torch.arange(w, device=image.device) + (1 + margin), max=w-1).long()
            outer_border_right = torch.clamp(torch.arange(w, device=image.device) + (2 + margin), max=w-1).long()

            right_diff = image[:, outer_border_right] - image[:, inner_border_right]
            
            right_widths_180 = torch.where(right_cond_180, margin + 1, right_widths_180)
            right_cond_180 = (outer_border_right < w) & (right_diff < 0) & right_cond_180

            right_widths_0 = torch.where(right_cond_0, margin + 1, right_widths_0)
            right_cond_0 = (outer_border_right < w) & (right_diff > 0) & right_cond_0

        
        edge_widths[mask_angle_180] = left_widths_180[mask_angle_180].float() + right_widths_180[mask_angle_180].float()
        edge_widths[mask_angle_0] = left_widths_0[mask_angle_0].float() + right_widths_0[mask_angle_0].float()
    
    return edge_widths







def count_values(tensor):
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    print(unique_elements, counts)

# @stop_watch
def _calculate_sharpness_metric(image, edges, edge_widths):
    # get the size of image

    # print(image.max(), image.min())
    # print(edges.max(), edges.min())
    # print(edge_widths.max(), edge_widths.min())

    h, w = image.shape

    # Initialize variables
    hist_pblur = torch.zeros(101, device=image.device)
    total_num_edges = 0

    # Block size
    BLOCK_HEIGHT, BLOCK_WIDTH = 64, 64
    THRESHOLD = 0.002
    BETA = 3.6
    WIDTH_JNB = torch.cat([5*torch.ones(51, device=image.device), 3*torch.ones(205, device=image.device)])

    # Number of blocks
    num_blocks_vertically = h // BLOCK_HEIGHT
    num_blocks_horizontally = w // BLOCK_WIDTH

    # Loop over blocks
    for i in range(num_blocks_vertically):
        for j in range(num_blocks_horizontally):
            rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
            cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))

            block_edges = edges[rows, cols]
            if is_edge_block(block_edges, THRESHOLD):
                block_widths = edge_widths[rows, cols]
                block_widths = block_widths[block_widths != 0]

                block_contrast = get_block_contrast(image[rows, cols])
                block_jnb = WIDTH_JNB[block_contrast]

                # Calculate the probability of blur detection at the edges detected in the block
                prob_blur_detection = 1 - torch.exp(-abs(block_widths / block_jnb) ** BETA)

                # Update the statistics using the block information
                hist_pblur += torch.histc(prob_blur_detection, bins=101, min=0, max=1)
                total_num_edges += prob_blur_detection.numel()

    # Normalize the pdf
    if total_num_edges > 0:
        hist_pblur /= total_num_edges

    # Calculate the sharpness metric
    sharpness_metric = torch.sum(hist_pblur[:64]).item()
    return sharpness_metric

def is_edge_block(block, threshold):
    # Decide whether the given block is an edge block
    return torch.count_nonzero(block) > (block.numel() * threshold)

def get_block_contrast(block):
    # Get the contrast of the given block
    return int(torch.max(block) - torch.min(block))
