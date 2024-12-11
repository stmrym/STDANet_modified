import matplotlib.pyplot as plt
import numpy as np
import torch

def matrix_imshow(data, save_name):
    '''
    data: np.array or torch.Tensor with shape (..., H, W)
    save_name: '*.png'
    '''
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
    assert len(data.shape) > 1

    # converting to (H, W)
    while len(data.shape) > 2:
        print(data.shape)
        data = data[0]
    print('end loop')
    print(data.shape)
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig(save_name)
    print(f'{save_name} saved.')
    plt.close()

