from models.STDAN_Stack import STDAN_Stack
from torchinfo import summary
from mmcv.cnn.utils import flops_counter

def main():
    model = STDAN_Stack()
    summary(
        model,
        input_size=(1,5, 3,256,256),
        col_names=['input_size', 'output_size', 'num_params'],
        depth = 1
    )

    flops_counter.get_model_complexity_info(model, input_shape=(5,3,256,256))



if __name__ == '__main__':
    main()