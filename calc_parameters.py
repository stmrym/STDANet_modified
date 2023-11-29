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


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


if __name__ == '__main__':
    main()

    import torch 
    x = torch.tensor([[[[1,2,3],
                      [4,5,6]]],
                      [[[7,8,9],
                       [10,11,12]]],
                       [[[13,14,15],
                         [16,17,18]]]]
    )
    x = x.reshape(1,3,1,2,3)
    srcs = x
    bs,t,c,h,w = srcs.shape
    
    masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
    
    valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
    
    src_flatten = []
    mask_flatten = []
    spatial_shapes = []
    for lv1 in range(t):
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
    print(spatial_shapes)
    x1 = spatial_shapes.new_zeros((1, ))
    print(x1)
    print(x1.shape)

    y1 = spatial_shapes.prod(1).cumsum(0)[:-1]
    print(spatial_shapes.prod(1))
    print(y1)
    print(y1.shape)
    z = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

    print(z)
    print(z.shape)
