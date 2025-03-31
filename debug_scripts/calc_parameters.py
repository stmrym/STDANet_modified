import sys
import os

# モジュールを含むディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.Stack import Stack
from torchinfo import summary
from mmcv.cnn.utils import flops_counter

def main():
    B = 1
    T = 3
    C = 3
    H = 1280
    W = 720


    model = Stack(  
                arch = 'ESTDAN', 
                use_stack = False, 
                n_sequence = 3, 
                in_channels = 3,
                n_feat = 32,
                out_channels = 3,
                n_resblock = 3,
                kernel_size = 5,
                # for ESTDAN
                use_cleaning=False,
                sobel_out_channels = 2,
                n_cleaning_blocks = 5,
                mid_channels = 32,
                device = 'cuda:0'
                )

    summary(
        model,
        input_size=(B,T,C,H,W),
        col_names=['input_size', 'output_size', 'num_params'],
        depth = 1
    )

    flops_counter.get_model_complexity_info(model, input_shape=(T,C,H,W))

    with open('ESTDAN_model_summary.txt', 'w') as writerfile:
        writerfile.write(repr(summary(
            model,
            input_size=(B,T,C,H,W),
            col_names=['input_size', 'output_size', 'num_params'],
            depth = 2
        )))



if __name__ == '__main__':
    main()

    
