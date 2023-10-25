import torch
from torchvision.ops import deform_conv2d
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
import os
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable


# digits = load_digits()
# input = torch.from_numpy(np.reshape(digits.data[0], (8,8))).float()

# input = input.unsqueeze(0)
# input = input.unsqueeze(0)

# print(input)

# kh, kw = 3, 3

# weight = torch.rand(1, 1, kh, kw)

# offset = torch.rand(1, 2*kh*kw, 6, 6)
# mask = torch.rand(1, kh*kw, 6, 6)

# # print(offset[0,:,0,0])


# out = deform_conv2d(input, offset, weight, mask=mask)

# # plt.matshow(input[0][0])
# # plt.matshow(out[0][0])
# # plt.show()


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
    mesh_flow[mag < minlength] = np.nan  # minlength以下をnanに置換

    u = mesh_flow[..., 0]
    v = mesh_flow[..., 1]

    return x, y, u, v

def adjust_ang(ang_min, ang_max):
    """Parameters
    input
    ang_min: start angle of degree
    ang_max: end angle of degree
    output
    unique_ang_min: angle after conversion to unique `ang_min`
    unique_ang_max: angle after conversion to unique `ang_max`
    """
    unique_ang_min = ang_min
    unique_ang_max = ang_max
    unique_ang_min %= 360
    unique_ang_max %= 360
    if unique_ang_min >= unique_ang_max:
        unique_ang_max += 360
    return unique_ang_min, unique_ang_max

def any_angle_only(mag, ang, ang_min, ang_max):
    """
    input
    mag: `cv2.cartToPolar` method `mag` reuslts
    ang: `cv2.cartToPolar` method `ang` reuslts
    ang_min: start angle of degree after `adjust_ang` function
    ang_max: end angle of degree after `adjust_ang` function
    output
    any_mag: array of replace any out of range `ang` with nan
    any_ang: array of replace any out of range `mag` with nan
    description
    Replace any out of range `mag` and `ang` with nan.
    """
    any_mag = np.copy(mag)
    any_ang = np.copy(ang)
    ang_min %= 360
    ang_max %= 360
    if ang_min < ang_max:
        any_mag[(ang < ang_min) | (ang_max < ang)] = np.nan
        any_ang[(ang < ang_min) | (ang_max < ang)] = np.nan
    else:
        any_mag[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[ang <= ang_max] += 360
    return any_mag, any_ang

def hsv_cmap(ang_min, ang_max, size):
    """
    input
    ang_min: start angle of degree after `adjust_ang` function
    ang_max: end angle of degree after `adjust_ang` function
    size: map px size
    output
    hsv_cmap_rgb: HSV color map in radial vector flow
    x, y, u, v: radial vector flow value
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    description
    Create a normalized hsv colormap between `ang_min` and `ang_max`.
    """
    # 放射状に広がるベクトル場の生成
    half = size // 2
    x = np.arange(-half, half+1, 1, dtype=np.float64)
    y = np.arange(-half, half+1, 1, dtype=np.float64)
    u, v = np.meshgrid(x, y)

    # HSV色空間の配列に入れる
    hsv = np.zeros((len(y), len(x), 3), dtype='uint8')
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    any_mag, any_ang = any_angle_only(mag, ang, ang_min, ang_max)
    hsv[..., 0] = 180*(any_ang - ang_min) / (ang_max - ang_min)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(any_mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv_cmap_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return hsv_cmap_rgb, x, y, u, v

prev_frame = cv2.imread('00000000.png')
next_frame = cv2.imread('00000001.png')

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

print(np.amax(flow[:,:,0]))
print(np.amin(flow[:,:,0]))

print(np.amax(flow[:,:,1]))
print(np.amin(flow[:,:,1]))

prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('prev_frame and prev2nextflow vector')
ax.imshow(prev_rgb)
x, y, u, v = flow_vector(flow=flow, spacing=10, margin=0, minlength=1)
ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=[0.0, 0.0, 1.0])

plt.savefig('arrow.png')


