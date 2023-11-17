from models.STDAN_Stack import STDAN_Stack
from torchinfo import summary

def main():
    model = STDAN_Stack()
    summary(
        model,
        input_size=(1,5, 3,256,256),
        col_names=['input_size', 'output_size', 'num_params'],
        depth = 1
    )



if __name__ == '__main__':
    main()