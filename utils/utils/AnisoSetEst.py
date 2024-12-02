import numpy as np
import torch
import torch.nn.functional as F
from utils import util
from utils.stop_watch import stop_watch

@stop_watch
def SVDCoherence(gmap):

    gx = np.real(gmap)
    gy = np.imag(gmap)

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


def AnisoSetEst(img, N):
    H, W = img.shape

    w = W // N
    h = H // N

    mp = np.zeros((h, w))
    alph = 0.001
    thresh = alph**(1 / (N**2 - 1))
    thresh = np.sqrt((1 - thresh) / (1 + thresh))


    for m in range(h):
        for n in range(w):
            AOI = img[N * m:N * (m + 1), N * n:N * (n + 1)]
            
            gx, gy = np.gradient(AOI)
            G = gx + 1j * gy
            co, _ = SVDCoherence(G)

            if co > thresh:
                mp[m, n] = 1

    return mp




def MetricQ(img, N, map):
    H, W = img.shape
    w = W//N
    h = H//N
    Q = 0

    for m in range(h):
        for n in range(w):
            if map[m, n] == 0:
                continue
            AOI = img[N * m:N * (m + 1), N * n:N * (n + 1)]
            gx, gy = np.gradient(AOI)

            G = gx + 1j * gy 
            co, s1 = SVDCoherence(G) 
            Q += co * s1

    Q /= (w * h)
    return Q


