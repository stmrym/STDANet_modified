import cv2
import numpy as np
import torch
import lpips
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union
from tqdm import tqdm
from utils import util
from losses.multi_loss import *


def calc_lpips():
    loss_fn_alex = lpips.LPIPS(net='alex')
    img0 = torch.rand(4,3,64,64)
    img1 = torch.rand(4,3,64,64)
    d = loss_fn_alex(img0, img1)
    print(d)



def radial_gradient(color,radii):
    colors=[]
    for r in radii:
        colorr=r*color+(1-r)*np.array([1,1,1])
        colors.append(colorr)
    return colors


def add_colors(colorwheel,cname,col,channel,key):
    if key==0:
        colorwheel[col:col+cname,channel]=255
        colorwheel[col:col+cname,(channel+1)%3]=np.floor(255*np.arange(cname)/cname)
    elif key==1:
        colorwheel[col:col+cname,channel]=255-np.floor(255*np.arange(cname)/cname)
        colorwheel[col:col+cname,(channel+1)%3]=255

def gen_colorwheel():
    RY,YG,GC,CB,BM,MR=15,6,4,11,13,6
    ncols=RY+YG+GC+CB+BM+MR
    colorwheel=np.zeros((ncols,3))
    col=0

    add_colors(colorwheel,RY,col,0,0)
    col=col+RY
    add_colors(colorwheel,YG,col,0,1)
    col=col+YG
    add_colors(colorwheel,GC,col,1,0)
    col=col+GC
    add_colors(colorwheel,CB,col,1,1)
    col=col+CB
    add_colors(colorwheel,BM,col,2,0)
    col=col+BM
    add_colors(colorwheel,MR,col,2,1)
    return colorwheel/255.0

def plot_colorwheel(colorwheel,steps=5, y_step=30):
    fig = plt.figure(figsize=(3,3), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111, projection='polar')
    ax.tick_params(bottom=False, left=False, right=False, top=False)
    N=colorwheel.shape[0] # N = 55
    x = np.linspace(0, 2*np.pi, N + 1)
    y = np.linspace(0, 1, y_step)

    for i in tqdm(range(N)):
        
        bins = np.linspace(x[i],x[i+1],steps)
        colors = np.linspace(colorwheel[i],colorwheel[(i+1)%N],steps)

        for j in range(steps):
            color = colors[j]
            line = radial_gradient(color, y)
            
            for k in range(len(y)):
                ax.scatter(bins[j], y[k], c = [line[k]])
    
    ax.set_theta_direction(-1) 
    ax.set_thetagrids([])
    ax.set_rgrids([])
    # ax.spines['polar'].set_visible(False)
    # ax.axis("off")
    plt.savefig('debug_results/colorwheel_2.png', transparent=True)


def make_color_wheel(bins: Optional[Union[list, tuple]] = None) -> np.ndarray:
    """Build a color wheel.

    Args:
        bins(list or tuple, optional): Specify the number of bins for each
            color range, corresponding to six ranges: red -> yellow,
            yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
            magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
            (see Middlebury).

    Returns:
        ndarray: Color wheel of shape (total_bins, 3).
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


def flow2rgb(flow: np.ndarray,
             color_wheel: Optional[np.ndarray] = None,
             unknown_thr: float = 1e6) -> np.ndarray:
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (float): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    # [0, 1]に正規化した強度分布 (h,w)
    rad = np.sqrt(dx**2 + dy**2)
    # [-1 ~ 1]の角度分布
    angle = np.arctan2(-dy, -dx) / np.pi
    # (h, w)
    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    # (h, w, 1)
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    # 要素ごとの積
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]

    # (h, w) True or False
    small_ind = rad <= 1
    
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0
    # (h,w,3) cは正規化したrgb値
    flow_map = np.uint8(flow_img * 255.)
    flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite('debug_results/twmp.png', flow_map)

    return flow_img
    


def flow2direction(flow: np.ndarray,
            color_wheel: Optional[np.ndarray] = None,
            unknown_thr: float = 1e6) -> np.ndarray:
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (float): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    # [0, 1]に正規化した強度分布 (h,w)
    rad = np.sqrt(dx**2 + dy**2)
    # [-1 ~ 1]の角度分布
    angle = np.arctan(dy/(dx + 1e-8)) * 2 / np.pi
        # (h, w)
    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    # (h, w, 1)
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    # 要素ごとの積
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]

    # (h, w) True or False
    small_ind = rad <= 1
    
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0
    # (h,w,3) cは正規化したrgb値
    flow_map = np.uint8(flow_img * 255.)
    flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite('debug_results/flow_half.png', flow_map)

    return flow_img



def orthogonal_edge_tensor():
    flow_path = './debug_results/27_00000007.npy'
    flow = np.load(flow_path)
    flow_tensor = torch.from_numpy(flow).clone().type(torch.cuda.FloatTensor).cuda()
    flow_tensor  = flow_tensor.permute(2,0,1).unsqueeze(dim = 0)
    print(flow_tensor.shape)

    img_path = './debug_results/27_00000007_out.png'
    # path = './exp_log/train/WO_Motion_2024-01-16T103421_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0350/output/GOPR0410_11_00/000198.png'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    out = torch.from_numpy(img_rgb/255).clone().type(torch.cuda.FloatTensor).cuda()
    out = out.permute(2,0,1).unsqueeze(dim = 0)
    print(out.device)


    util.save_edge(
        savename = './debug_results/27_00000007_w_amp_3.png', 
        out_image = out,
        flow_tensor = flow_tensor,
        key = 'abs_weight',
        edge_extraction_func = orthogonal_edge_extraction)






def orthogonal_edge_numpy():

    flow_path = './debug_results/000198.npy'
    flow = np.load(flow_path)

    img_path = './debug_results/000198.png'
    # img_path = './exp_log/train/WO_Motion_2024-01-16T103421_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0350/output/GOPR0410_11_00/000198.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # img = np.full((512, 512), 0, dtype=np.uint8)
    # cv2.circle(img, (256,256), 150, (255,255), thickness=-1)
    # cv2.imwrite('./debug_results/input_gray.png', img)

    # img = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #      ]
    # , dtype=np.float64)

    sobel_x_kernel = np.array(
        [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]
        )

    sobel_y_kernel = np.array(
        [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
        ]
        )

    sobel_x = cv2.filter2D(img, cv2.CV_64F, sobel_x_kernel)
    sobel_y= cv2.filter2D(img, cv2.CV_64F, sobel_y_kernel)

    print(sobel_x)
    print(sobel_y)

    amp = np.sqrt(sobel_x**2 + sobel_y**2)
    # [0, 1] normalized
    amp /= np.max(amp)

    edge_direction = np.stack([sobel_x, sobel_y], axis=-1)

    if flow.shape != edge_direction.shape:
        flow = cv2.resize(flow, edge_direction.shape[1::-1])

    prod = np.multiply(flow[:,:,0], edge_direction[:,:,0]) + np.multiply(flow[:,:,1], edge_direction[:,:,1])
    abs_prod = np.abs(prod)
    # [0, 1] normalized
    # abs_prod /= np.max(abs_prod)

    orthogonal_edge = np.multiply(abs_prod, amp)


    amp = np.uint8(amp/np.max(amp) * 255.)
    abs_prod = np.uint8(abs_prod/np.max(abs_prod) * 255.)
    orthogonal_edge = np.uint8(orthogonal_edge/np.max(orthogonal_edge) * 255.)

    cv2.imwrite('./debug_results/000198_amp.png', amp)
    cv2.imwrite('./debug_results/000198_wamp.png', abs_prod)
    # cv2.imwrite('./debug_results/27_00000007_orth_edge.png', orthogonal_edge)




    # # angle = flow2rgb(edge_flow)
    # edge_half_angle = flow2direction(edge_direction)



    # angle = np.arctan(sobel_y/sobel_x)/ np.pi
    # angle = (angle + 1)/2
    # print(amp)


    # fig, ax = plt.subplots(tight_layout=True, dpi=200)
    # im = ax.imshow(edge_half_angle, vmin=-90, vmax=90, cmap='hsv', aspect='equal')

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("top", size="5%", pad=0.3)

    # fig.colorbar(im, cax=cax, orientation='horizontal')
    # plt.savefig('./debug_results/angle_2.png')

    # with open('./debug_results/angle.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(angle)

    # cv2.imwrite('./debug_results/output_gray.png', img)
    # cv2.imwrite('./debug_results/output_x.png', sobel_x)
    # cv2.imwrite('./debug_results/output_y.png', sobel_y)

if __name__ == '__main__':
    # orthogonal_edge_numpy()
    calc_lpips()