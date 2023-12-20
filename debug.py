import torch
from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from models.STDAN_Stack import STDAN_Stack


def main():

    B = 7
    T = 5
    C = 3
    H = 256
    W = 256
    input = torch.zeros(B, T, C, H, W)
    deblurnet = STDAN_RAFT_Stack().to('cuda:0')
    input = input.to('cuda:0')
    deblurnet(input)
    







if __name__ == '__main__':
    main()

