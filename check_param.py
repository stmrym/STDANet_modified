import glob
import os
import pandas as pd
import torch
from models.ESTDAN_Stack import ESTDAN_Stack



def get_weights(path, multi_file = True):
    if multi_file:
        weights = sorted(glob.glob(os.path.join(path, 'ckpt-epoch-*.pth.tar')))
    else:
        weights = [path]
    return weights


if __name__ == '__main__':

    path = './exp_log/train/F_2024-03-06T124456_ESTDAN_Stack_BSD_3ms24ms_GOPRO/checkpoints'
    weights = get_weights(path, multi_file = True)

    index_list = [s.split('ckpt-epoch-')[1].split('.pth.tar')[0] for s in weights]
    row_list = ['x11', 'x12', 'x13', 'x21', 'x22', 'x23', 'x31', 'x32', 'x33',
                'y11', 'y12', 'y13', 'y21', 'y22', 'y23', 'y31', 'y32', 'y33', 'b1', 'b2']

    df_tensor_list = []
    for weight in weights:
        print(len(df_tensor_list))
        checkpoint = torch.load(weight, map_location='cpu')
        param_tensor = torch.zeros((0,))
        for name, param in checkpoint['deblurnet_state_dict'].items():
            if 'edge_extractor' in name:
                if len(param.shape) != 1:
                    param = param.view(-1)
                param_tensor = torch.cat((param_tensor, param))
        # (x11, x12, x13, x21, x22, x23, x31, x32, x33, 
        #  y11, y12, y13, y21, y22, y23, y31, y32, y33, b1, b2)
        print(param_tensor)
        exit()
        df_tensor_list.append(param_tensor.unsqueeze(0))
    
    df_tensors = torch.cat(df_tensor_list, dim=0)
    df = pd.DataFrame(df_tensors.numpy())
    df.columns = row_list
    df.index = index_list

    print(df)
    df.to_csv('param.csv')



