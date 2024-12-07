import cv2
import torch
import numpy as np
from bm3d import bm3d
import sys
import os
sys.path.append(os.path.dirname(__file__))

from stop_watch import stop_watch

@stop_watch
def denoise(img):
    THRESHOLD = 0.01
    LOW = 0.0
    HIGH = 0.5
    MIN_STEP = 0.0005

    cont = True
    result = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    denoised, err = bm3d_twocolor(img, LOW)
    result.append([LOW, err])
    if err <= THRESHOLD:
        cont = False

    if cont:
        denoised, err = bm3d_twocolor(img, HIGH)
        result.append([HIGH, err])
        if err > THRESHOLD:
            cont = False

    cur_low = LOW
    cur_high = HIGH
    while cont:
        cur = (cur_low + cur_high) * 0.5
        denoised, err = bm3d_twocolor(img, cur)
        result.append([cur, err])

        if err <= THRESHOLD:
            cur_high = cur
        else:
            cur_low = cur

        print(cur_low, MIN_STEP, cur_high)
        if (cur_low + MIN_STEP >= cur_high):
            idx = np.abs(np.array(result)[:, 0] - cur_high).argmin()
            assert idx is not None

            denoised, err = bm3d_twocolor(img, cur_high)
            result.append([cur_high, err])
            cont = False

    denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
    return denoised

@stop_watch
def bm3d_twocolor(img, noise_level):
    if noise_level > 1e-6:
        denoised = bm3d(img, sigma_psd=noise_level * 255).astype(np.float32)
    else:
        denoised = img
    
    _, _, err = two_color(denoised)
    err = (np.mean(err**0.8))**(1 / 0.8)

    return denoised, err



def im2col(A, block_size, stepsize=1):
    '''
    A: (H, W) image [0, 1]
    block_size: patch size
    '''

    M, N = A.shape
    b0, b1 = block_size
   
    # Calculate the output dimensions
    col_extent = N - b1 + 1
    row_extent = M - b0 + 1

    # Create list of starting block indices
    start_idx = np.arange(b0)[:, None] * N + np.arange(b1)

    # Generate depth indices for column offsets
    offset_idx = np.arange(0, row_extent, stepsize)[:, None] * N + np.arange(0, col_extent, stepsize)

    # Get all actual indices & index into the input array for final output
    out = np.take(A, start_idx.ravel('F')[:, None] + offset_idx.ravel('F'))

    # Reshape to make each column a block
    out_shape = (b0 * b1, -1)
    out = out.reshape(out_shape)

    return out



def init_centers(r_col, g_col, b_col, patch_size):
    idx = np.random.randint(0, patch_size[0]*patch_size[1], size=(1, r_col.shape[1]))

    rc = [None] * 2
    gc = [None] * 2
    bc = [None] * 2

        
    c_idx = np.ravel_multi_index((np.arange(len(idx[0])), idx[0]), r_col.T.shape)


    rc[0] = r_col.ravel('F')[c_idx]
    gc[0] = g_col.ravel('F')[c_idx]
    bc[0] = b_col.ravel('F')[c_idx]

    diff = (r_col - rc[0])**2 + (g_col - gc[0])**2 + (b_col - bc[0])**2
    nonzero_num = np.sum(diff > 1e-12, axis=0)
    s_idx = np.argsort(diff, axis=0)[::-1]

    s_sub2ind = np.ravel_multi_index((np.arange(len(nonzero_num)), np.maximum(np.ceil(nonzero_num * 0.5).astype(int), 1) - 1), s_idx.T.shape)
    idx = s_idx.ravel('F')[s_sub2ind]


    c_idx = np.ravel_multi_index((np.arange(len(idx)), idx), r_col.T.shape)

    rc[1] = r_col.ravel('F')[c_idx]
    gc[1] = g_col.ravel('F')[c_idx]
    bc[1] = b_col.ravel('F')[c_idx]
    
    return rc, gc, bc




def two_color(img):
    assert img.shape[2] == 3
    
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    patch_size = (5, 5)
    margin = (patch_size[0] // 2, patch_size[1] // 2)

    r_col = im2col(r, patch_size)
    g_col = im2col(g, patch_size)
    b_col = im2col(b, patch_size)
    
    r = r[margin[0]: -margin[0], margin[1]: -margin[1]]
    g = g[margin[0]: -margin[0], margin[1]: -margin[1]]
    b = b[margin[0]: -margin[0], margin[1]: -margin[1]]
    img = img[margin[0]: -margin[0], margin[1]: -margin[1], :]

    r_centers, g_centers, b_centers = init_centers(r_col, g_col, b_col, patch_size)

    diff = [None, None]
    max_iter = 10
    for iter in range(max_iter):
        for k in range(2):
            diff[k] = (r_col - r_centers[k])**2 + (g_col - g_centers[k])**2 + (b_col - b_centers[k])**2
            
        map = (diff[0] <= diff[1]).astype(np.float64)

        for k in range(2):
            map_sum = np.sum(map, axis=0)
            map_sum[map_sum < 1e-10] = 1e+10

            norm_coef = 1.0 / map_sum
            r_centers[k] = np.sum(r_col * map, axis=0) * norm_coef
            g_centers[k] = np.sum(g_col * map, axis=0) * norm_coef
            b_centers[k] = np.sum(b_col * map, axis=0) * norm_coef
        
            map = 1.0 - map
        
        diff1 = (r_centers[0] - r.ravel('F'))**2 + (g_centers[0] - g.ravel('F'))**2 + (b_centers[0] - b.ravel('F'))**2
        diff2 = (r_centers[1] - r.ravel('F'))**2 + (g_centers[1] - g.ravel('F'))**2 + (b_centers[1] - b.ravel('F'))**2
        map = diff1 > diff2
        
        tmp = r_centers[0][map]
        r_centers[0][map] = r_centers[1][map]
        r_centers[1][map] = tmp

        tmp = g_centers[0][map]
        g_centers[0][map] = g_centers[1][map]
        g_centers[1][map] = tmp
        
        tmp = b_centers[0][map]
        b_centers[0][map] = b_centers[1][map]
        b_centers[1][map] = tmp

    center1 = np.zeros_like(img)
    center1[:, :, 0] = r_centers[0].reshape(r.shape)
    center1[:, :, 1] = g_centers[0].reshape(g.shape)
    center1[:, :, 2] = b_centers[0].reshape(b.shape)
        
    center2 = np.zeros_like(img)
    center2[:, :, 0] = r_centers[1].reshape(r.shape)
    center2[:, :, 1] = g_centers[1].reshape(g.shape)
    center2[:, :, 2] = b_centers[1].reshape(b.shape)

    diff = center2 - center1
    len = np.sqrt(np.sum(diff**2, axis=2))
    dir = diff / (len[..., np.newaxis] + 1e-12)

    diff = img - center1
    proj = np.sum(diff * dir, axis=2)
    dist = diff - dir * proj[..., np.newaxis]
    err = np.sqrt(np.sum(dist**2, axis=2))

    return center1, center2, err

