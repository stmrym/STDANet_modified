import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import math
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from mmflow.datasets import visualize_flow
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from losses.multi_loss import *

def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, torch.Tensor):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.to('cpu')
    if isinstance(img2, torch.Tensor):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.to('cpu')
    mse = torch.mean((img1 / rgb_range - img2 / rgb_range) ** 2,[1,2,3])
    """ if mse == 0:
        return 100 """
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse)).mean()


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img
def ssim_calculate(x, y, val_range=255.0):
    ssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=val_range)
    return ssim


def flow_vector(flow, spacing, margin, minlength):
    """Parameters:
    input
    flow: motion vectors 3D-array
    spacing: pixel spacing of the flow
    margin: pixel margins of the flow
    minlength: minimum pixels to leave as flow
    output
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    """
    h, w, _ = flow.shape

    x = np.arange(margin, w - margin, spacing, dtype=np.int64)
    y = np.arange(margin, h - margin, spacing, dtype=np.int64)

    mesh_flow = flow[np.ix_(y, x)]
    mag, _ = cv2.cartToPolar(mesh_flow[..., 0], mesh_flow[..., 1])
    mesh_flow[mag < minlength] = np.nan  # replace under minlength to nan

    u = mesh_flow[..., 0]
    v = mesh_flow[..., 1]

    return x, y, u, v


def save_edge(savename:str, out_image:torch.tensor, flow_tensor:torch.tensor, key:str, edge_extraction_func):
    
    output_dict = edge_extraction_func(out_image, flow_tensor)

    # key: 'weighted', 'edge', or 'flow_magnitude'
    output = output_dict[key]

    # Min-max normalization
    # if torch.max(output) != torch.min(output):
        # output = (output - torch.min(output))/(torch.max(output) - torch.min(output))

    # Edge enhancing
    # if key in ['edge', 'weighted']:
    #     output *= 5
    
    # print(f'{torch.max(output)} {torch.min(output)}')

    torchvision.utils.save_image(output, savename)


def save_hsv_flow_from_outflow(save_dir, flow_type='out_flow', save_vector_map=False):

    seqs = sorted([f for f in os.listdir(os.path.join(save_dir, flow_type + '_npy')) if os.path.isdir(os.path.join(save_dir, flow_type + '_npy', f))])
    tqdm_seqs = tqdm(seqs)
    tqdm_seqs.set_description(f'[SAVE]')

    for seq in tqdm_seqs:
        npy_files = sorted(glob.glob(os.path.join(save_dir, flow_type + '_npy', seq, '*.npy')))
        out_flows = []
        names = []
        for npy_file in npy_files:
            npy = np.load(npy_file)
            H, W, _ = npy.shape
            npy = cv2.resize(npy, (W*4, H*4))
            out_flows.append(npy)
            names.append(os.path.splitext((os.path.basename(npy_file)))[0])

        if save_vector_map == True:
            firstLoop = True
            for out_flow in out_flows:  # get vector_max for each seq       
                _, _, u, v = flow_vector(flow=out_flow, spacing=10, margin=0, minlength=1)  # flow.shape must be (H, W, 2)
                vector_mag = np.nanmax(np.sqrt(pow(u,2)+pow(v,2)))

                if firstLoop == True:
                    vector_amax = vector_mag
                    firstLoop = False
                else:
                    if vector_amax < vector_mag:
                        vector_amax = vector_mag
        

        for img_name, out_flow in zip(names, out_flows):
            
            ############################
            # saving flow_hsv using mmcv
            ############################
            
            flow_map = visualize_flow(out_flow, None)
            # visualize_flow return flow map with RGB order
            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)

            if os.path.isdir(os.path.join(save_dir, flow_type, seq)) == False:
                os.makedirs(os.path.join(save_dir, flow_type, seq), exist_ok=True)
            
            cv2.imwrite(os.path.join(save_dir, flow_type, seq, img_name + '.png'), flow_map) 

            if save_vector_map == True:
                ####################
                # saving flow_vector
                ####################

                fig, ax = plt.subplots(figsize=(8,6), dpi=350)
                ax.axis("off")
                output_image = cv2.imread(os.path.join(save_dir, 'output', seq, img_name + '.png'), cv2.IMREAD_GRAYSCALE)

                ax.imshow(output_image.astype(np.uint8),cmap='gray', alpha=0.8)
                
                x, y, u, v = flow_vector(flow=out_flow, spacing=10, margin=0, minlength=5)  # flow.shape must be (H, W, 2)
                im = ax.quiver(x, y, u/np.sqrt(pow(u,2)+pow(v,2)),v/np.sqrt(pow(u,2)+pow(v,2)),np.sqrt(pow(u,2)+pow(v,2)), cmap='jet', angles='xy', scale_units='xy', scale=0.1)
                
                divider = make_axes_locatable(ax) # get AxesDivider
                cax = divider.append_axes("right", size="5%", pad=0.1) # make new axes
                cb = fig.colorbar(im, cax=cax)
                cb.mappable.set_clim(0, vector_amax)

                forward_path = os.path.join(save_dir, 'flow_forward', seq)
                if os.path.isdir(forward_path) == False:
                    os.makedirs(forward_path)
                plt.savefig(os.path.join(forward_path, img_name + '.png'), bbox_inches = "tight")



def save_hsv_flow(save_dir:str, seq:str, img_name:str, out_flow):
            
    ############################
    # saving flow_hsv using mmcv
    ############################
    
    flow_map = visualize_flow(out_flow, None)
    # visualize_flow return flow map with RGB order
    flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)

    if os.path.isdir(os.path.join(save_dir +  '_out_flow', seq)) == False:
        os.makedirs(os.path.join(save_dir + '_out_flow', seq), exist_ok=True)
    
    cv2.imwrite(os.path.join(save_dir + '_out_flow', seq, img_name + '.png'), flow_map) 

