import numpy as np
from scipy.ndimage import convolve
from pymlfunc import normxcorr2


import sys
import os
sys.path.append(os.path.dirname(__file__))

from debug_util import matrix_imshow

def align(image, ref, return_images=False):

    margin = 75
    
    # テンプレートの抽出
    template = image[margin:-margin, margin:-margin, :]
    template_size = template.shape
    template_size = np.array(template_size) + (np.mod(template_size, 2) - 1)
    template_size = template_size[0:2]
    template = template[:template_size[0], :template_size[1], :]

    # NCCの計算
    ref_size = ref.shape
    ncc_size = tuple(np.array(ref_size[0:2]) + np.array(template_size) - 1)
    ncc = np.zeros(ncc_size)

    for k in range(3):
        ncc += normxcorr2(template[:, :, k], ref[:, :, k])


    ncc_margin = np.floor(np.array(template_size) / 2).astype(int)
    ncc = ncc[ncc_margin[0]:-ncc_margin[0], ncc_margin[1]:-ncc_margin[1]]

    # dy: row, dx: col
    dx, dy = np.unravel_index(np.argmax(ncc), ncc.T.shape)
    dy = dy - np.floor(template_size[0] / 2) - margin
    dx = dx - np.floor(template_size[1] / 2) - margin
    dy, dx = int(dy), int(dx)

    if return_images:
        if dy < 0:
            height = min(image.shape[0] + dy, ref.shape[0])
            image = image[-dy:height-dy, :, :]
            ref = ref[:height, :, :]
        else:
            height = min(image.shape[0], ref.shape[0] - dy)
            image = image[:height, :, :]
            ref = ref[dy:height+dy, :, :]
        if dx < 0:
            width = min(image.shape[1] + dx, ref.shape[1])
            image = image[:, -dx:width-dx, :]
            ref = ref[:, :width, :]
        else:
            width = min(image.shape[1], ref.shape[1] - dx)
            image = image[:, :width, :]
            ref = ref[:, dx:width+dx, :]
    else:
        return dx, dy

    return image, ref



def make_gaussian(len):
    r = len // 2
    assert r + r + 1 == len
    
    sigma = r / 3.0
    g = np.exp(-np.square(np.arange(-r, r + 1)) / (2 * sigma * sigma))
    g = g / g[r]
    return g


def apply_filter(emask, filter_width):
    # フィルタの形状を定義
    filter_shape = (filter_width, filter_width)
    
    # フィルタの適用
    if emask.ndim == 2:
        # 2次元の場合
        emask = convolve(emask, np.ones(filter_shape), mode='constant') > 0
    elif emask.ndim == 3:
        # 3次元の場合、各チャネルごとに処理
        for i in range(emask.shape[2]):
            emask[:, :, i] = convolve(emask[:, :, i], np.ones(filter_shape), mode='constant') > 0
    return emask



def grad_ring(latent, ref):
    # assert latent.dtype == np.float64
    # assert ref.dtype == np.float64
    assert ref.shape[2] == latent.shape[2]

    g = make_gaussian(15)
    assert g.ndim == 1

    result = np.zeros_like(latent)
    for c in range(ref.shape[2]):
        tdx, tdy = np.gradient(ref[:, :, c])
        ldx, ldy = np.gradient(latent[:, :, c])

        rx = np.abs(ldx) - convolve(np.abs(tdx), g[:, np.newaxis], mode='constant')
        ry = np.abs(ldy) - convolve(np.abs(tdy), g[np.newaxis, :], mode='constant')
        rx = np.maximum(rx, 0)
        ry = np.maximum(ry, 0)

        result[:, :, c] = np.sqrt(rx**2 + ry**2)

    gx = np.zeros_like(ref)
    gy = np.zeros_like(ref)
    for c in range(ref.shape[2]):
        gx[:, :, c], gy[:, :, c] = np.gradient(ref[:, :, c])
    g = np.sqrt(gx**2 + gy**2)

    filter_width = int(np.floor(max(max(latent.shape) / 200, 1)))

    emask = g > 0.03
    emask = apply_filter(emask, filter_width)
    result[emask] = 0.0

    return result