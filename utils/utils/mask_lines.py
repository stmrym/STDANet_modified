import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.ndimage import convolve


def mask_points(mask, xs, ys):
    indices = np.ravel_multi_index((xs, ys), mask.T.shape, mode='clip') # マスクを更新 
    mask.ravel('F')[indices] = True
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




