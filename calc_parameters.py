from models.STDAN_Stack import STDAN_Stack
from torchinfo import summary
from mmcv.cnn.utils import flops_counter

def main():
    B = 1
    T = 5
    C = 3
    H = 256
    W = 256
    model = STDAN_Stack()
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
            depth = 20
        )))



if __name__ == '__main__':
    main()

    
