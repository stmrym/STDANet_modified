import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

def compute_ncc(img, ref, img_margin):

    assert len(img.shape) == 2, 'Image should have a single channel.'

    template = ref[img_margin:-img_margin, img_margin:-img_margin]
    ncc = np.ones((img_margin * 2 + 1, img_margin * 2 + 1)) * 100
    ncc_abs = np.ones((img_margin * 2 + 1, img_margin * 2 + 1))

    img_mask = mask_lines(img)
    # ref_mask = mask_lines(ref)
    ref_mask = img_mask.copy()


    # print(img_mask)
    # print(img_mask.shape)
    # print(img_mask.max(), img_mask.min())
    # img_mask_np = np.clip(img_mask.astype(np.float32)*255, 0, 255).astype(np.uint8)
    # cv2.imwrite('edge_mask_cpu.png', img_mask_np)
    # exit()

    t_mask = ref_mask[img_margin:-img_margin, img_margin:-img_margin]

    dx, dy = np.gradient(img) 
    tdx, tdy = np.gradient(template)

    dx[img_mask] = 0 
    dy[img_mask] = 0 
    tdx[t_mask] = 0 
    tdy[t_mask] = 0

    ncc_dx = xcorr2_fft(tdx, dx)
    ncc_dy = xcorr2_fft(tdy, dy)


    ncc_dx = ncc_dx[tdx.shape[0]-1:, tdx.shape[1]-1:]
    ncc_dy = ncc_dy[tdy.shape[0]-1:, tdy.shape[1]-1:]

    ncc_dx = ncc_dx[:img_margin * 2 + 1, :img_margin * 2 + 1]
    ncc_dy = ncc_dy[:img_margin * 2 + 1, :img_margin * 2 + 1]

    ncc_dx = ncc_dx / ncc_dx[img_margin, img_margin]
    ncc_dy = ncc_dy / ncc_dy[img_margin, img_margin]

    ncc_dx_abs = np.abs(ncc_dx)
    ncc_dy_abs = np.abs(ncc_dy)

    mask = ncc_dx_abs < ncc_abs

    ncc[mask] = ncc_dx[mask]
    ncc_abs[mask] = ncc_dx_abs[mask]

    mask = ncc_dy_abs < ncc_abs
    ncc[mask] = ncc_dy[mask]
    ncc_abs[mask] = ncc_dy_abs[mask]

    return ncc


def mask_lines(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(img.shape, dtype=bool)
    e = canny(img)

    filter = np.ones((3, 3))
    for _ in range(20):
        cur_mask = mask_line(e)
    
        e[cur_mask] = False

        cur_mask = convolve(cur_mask, filter, mode='constant', cval=0.0)
        cur_mask = cur_mask > 0

        mask[cur_mask] = True

    return mask



def mask_line(e):

    # H, theta, rho = hough_line(e, theta=np.linspace(-np.pi/2, np.pi/2, 360))
    # P = hough_line_peaks(H, theta, rho, num_peaks=2, threshold=0.2 * np.max(H))
    lines = probabilistic_hough_line(e, threshold=10, line_length=20, line_gap=8)
    
    len_max = int(np.ceil(max(e.shape) * 3))
    mask = np.zeros(e.shape, dtype=bool)

    for line in lines:
        if line is None:
            continue
        xy = np.array([line[0], line[1]])

        xs = np.linspace(xy[0, 0], xy[1, 0], len_max) 
        ys = np.linspace(xy[0, 1], xy[1, 1], len_max)

        mask = mask_points(mask, np.floor(xs).astype(int), np.floor(ys).astype(int)) 
        mask = mask_points(mask, np.floor(xs).astype(int), np.ceil(ys).astype(int)) 
        mask = mask_points(mask, np.ceil(xs).astype(int), np.floor(ys).astype(int)) 
        mask = mask_points(mask, np.ceil(xs).astype(int), np.ceil(ys).astype(int))

    return mask


def mask_points(mask, xs, ys):

    indices = np.ravel_multi_index((xs, ys), mask.T.shape) # マスクを更新 
    # mask.ravel('F')[indices] = True
    
    for x, y in zip(xs, ys):
        mask[y,x] = True
    
    return mask


def xcorr2_fft(a, b):
    b_rot = np.rot90(np.conj(b), 2) 
    result = fftconvolve(a, b_rot, mode='full') 
    return result