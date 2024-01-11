import torch
from models.STDAN_RAFT_Stack import STDAN_RAFT_Stack
from models.STDAN_Stack import STDAN_Stack
from utils import util
import cv2

def main():

    # B = 7
    # T = 5
    # C = 3
    # H = 256
    # W = 256
    # input = torch.zeros(B, T, C, H, W)
    # deblurnet = STDAN_RAFT_Stack().to('cuda:0')
    # input = input.to('cuda:0')
    # deblurnet(input)

    image = cv2.imread('./debug_results/027.00000012_out.png', cv2.IMREAD_GRAYSCALE)

    kernel_size = 5                                                         # カーネルサイズの設定
    sigma = 0                                                               # sigmaの設定
    # blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)    # ガウシアンフィルターの適用
    # ラプラシアン フィルターを適用
    edge = cv2.Laplacian(image, cv2.CV_8U, ksize=3)
    print(edge)
    edge *= 3
    cv2.imwrite('./debug_results/027.00000012_out_edge.png',edge)
    







if __name__ == '__main__':
    main()

