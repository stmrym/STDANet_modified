from models.Stack import Stack
from torchinfo import summary
from mmcv.cnn.utils import flops_counter
from config.config_debug import cfg

def main():
    B = 1
    T = 3
    C = 3
    H = 256
    W = 256


    model = Stack(  
                network_arch = 'ESTDAN_v2', 
                use_stack = False, 
                n_sequence = 3, 
                in_channels = 3,
                n_feat = 32,
                out_channels = 3,
                n_resblock = 3,
                kernel_size = 5,
                sobel_out_channels = 2,
                device = 'cuda:0'
                )

    summary(
        model,
        input_size=(B,T,C,H,W),
        col_names=['input_size', 'output_size', 'num_params'],
        depth = 1
    )

    flops_counter.get_model_complexity_info(model, input_shape=(T,C,H,W))

    with open('model_summary.txt', 'w') as writerfile:
        writerfile.write(repr(summary(
            model,
            input_size=(B,T,C,H,W),
            col_names=['input_size', 'output_size', 'num_params'],
            depth = 2
        )))



if __name__ == '__main__':
    main()

    
