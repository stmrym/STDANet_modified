from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from torchinfo import summary
from mmcv.cnn.utils import flops_counter
from config.config_debug import cfg

def main():
    B = 1
    T = 5
    C = 3
    H = 256
    W = 256

    model = STDAN_RAFT_Stack(cfg=cfg)
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

    
