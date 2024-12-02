import numpy as np
import torch
import torch.nn.functional as F

def mean_norm(v, p):
    return np.mean(np.abs(v)**p)**(1/p)


def my_sd(x, p):
    avg = np.mean(x)
    sd = np.mean(np.abs(x - avg)**p) ** (1.0/p)
    return sd

