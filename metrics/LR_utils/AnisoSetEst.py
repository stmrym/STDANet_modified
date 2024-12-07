import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from stop_watch import stop_watch

# @stop_watch
def SVDCoherence(gx, gy):

    gxvect = gx.ravel('F')
    gyvect = gy.ravel('F')

    grad = np.vstack((gxvect, gyvect)).T
    U, S, Vt = np.linalg.svd(grad, full_matrices=False)

    s1 = float(S[0])
    s2 = float(S[1])

    if s1 + s2 == 0:
        co = 0.0
    else:
        co = np.abs(s1 - s2) / (s1 + s2)

    return co, np.abs(s1)

def MetricQ(img, N):
    H, W = img.shape

    w = W // N
    h = H // N

    # mp = np.zeros((h, w))
    alph = 0.001
    thresh = alph**(1 / (N**2 - 1))
    thresh = np.sqrt((1 - thresh) / (1 + thresh))

    Q = 0
    for m in range(h):
        for n in range(w):
            AOI = img[N * m:N * (m + 1), N * n:N * (n + 1)]
            
            # gx, gy = np.gradient(AOI)
            gy, gx = np.gradient(AOI)

            # G = gx + 1j * gy
            co, s1 = SVDCoherence(gx, gy)

            if co > thresh:
                # mp[m, n] = 1
                Q += co * s1

    Q /= (w*h)
    return Q




