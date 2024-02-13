import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union

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
    cv2.imwrite('debug_results/flow_4.png', flow_map)

    return flow_img
    



path = 'exp_log/train/2024-01-29T051821_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0350/out_flow_npy/027/00000024.npy'
flow = np.load(path)

# flow2rgb(flow)

img = cv2.imread('../dataset/GOPRO_Large/test/GOPR0854_11_00/sharp/000008.png', cv2.IMREAD_GRAYSCALE)
print(type(img[0,0]))

# img = np.full((512, 512), 0, dtype=np.uint8)
# cv2.circle(img, (256,256), 150, (255,255), thickness=-1)


sobel_x_kernel = np.array(
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]]
    )

sobel_y_kernel = np.array(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]
    ]
    )


sobel_x = cv2.filter2D(img, cv2.CV_64F, sobel_x_kernel)
sobel_y= cv2.filter2D(img, cv2.CV_64F, sobel_y_kernel)
amp = np.sqrt(sobel_x**2 + sobel_y**2)


edge_flow = np.stack([sobel_x, sobel_y], axis=-1)

angle = flow2rgb(edge_flow)


# angle = np.arctan2(-sobel_y, -sobel_x)/ np.pi
# angle = (angle + 1)/2
# print(amp)


fig, ax = plt.subplots(tight_layout=True, dpi=200)
im = ax.imshow(angle, vmin=-180, vmax=180, cmap='hsv', aspect='equal')

divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="5%", pad=0.3)

fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig('./debug_results/angle_2.png')

# with open('./debug_results/angle.csv', 'w', newline='') as file:
    # writer = csv.writer(file)
    # writer.writerows(angle)

# cv2.imwrite('./debug_results/output_gray.png', img)
# cv2.imwrite('./debug_results/output_x.png', sobel_x)
# cv2.imwrite('./debug_results/output_y.png', sobel_y)