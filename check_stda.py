import os
import cv2
import numpy as np
import torch
from models.Stack import Stack
from models.submodules import warp
import torch.nn.functional as F
from torchvision.utils import save_image
from typing import List, Tuple
from models.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION


def read_image_from_filename(filename_list: List[str]) -> List[np.ndarray]:

    image_list = []
    for filename in filename_list:
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list


def convert_input_tesnor_list(input_list: List[np.ndarray]) -> List[torch.Tensor]:
    # Convert np.array list to input tensor list

    # [np.arrray(h, w, c), ...] -> torch.Tensor(n, h, w, c)
    inputs = np.stack(input_list)
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).cuda()
    inputs = inputs.float() / 255
    # (1, n, c, h, w)
    input_tensor = torch.unsqueeze(inputs, dim = 0)

    num_frame = input_tensor.shape[1]
    input_tensor_list = []
    for i in range(0, num_frame - 4):
        input_tensor_list.append(input_tensor[:, i:i+5, :, :, :])

    return input_tensor_list


def vis_flow():

    img1 = cv2.imread('/mnt/d/results/20240529/2024-05-13T191746_STDAN_GOPRO_GOPRO_raw epoch 1200/00034.png')
    img2 = cv2.imread('./exp_log/test/debug_/flow_out_flow/001/00034.png')
    img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)

    alpha = 0.5
    img = alpha * img1.astype(np.float32) + (1-alpha) *img2.astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite('/mnt/d/results/20240529/2024-05-13T191746_STDAN_GOPRO_GOPRO_raw epoch 1200/blend.png', img)

    # filename_list = [
    #     '../dataset/chronos/test/0306-222113/sharp/00032.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00033.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00034.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00035.png',
    #     '../dataset/chronos/test/0306-222113/sharp/00036.png'
    # ]
    # image_list = read_image_from_filename(filename_list)
    # gt_seq = convert_input_tesnor_list(image_list)[0]
    

    input_seq = torch.load('./debug_results/input.pt')
    # gt_seq = torch.load('./debug_results/gt_seq.pt')
    flow_forward = torch.load('./debug_results/flow_forwards.pt')
    flow_backward = torch.load('./debug_results/flow_backwards.pt')
    

    flow_forward_npy = flow_forward.detach().cpu().numpy()
    np.savetxt('./debug_results/forward_00.csv', flow_forward_npy[0,0,0,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_01.csv', flow_forward_npy[0,0,1,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_10.csv', flow_forward_npy[0,1,0,:,:], fmt='%.3f', delimiter=',')
    np.savetxt('./debug_results/forward_11.csv', flow_forward_npy[0,1,1,:,:], fmt='%.3f', delimiter=',')



    b,t,c,h,w = input_seq.shape
    down_simple_gt = F.interpolate(input_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
    # frames = down_simple_gt[:,[1,2,3],:,:,:]
    frames = down_simple_gt[:,[0,1,2],:,:,:]
    n, t, c, h, w = frames.size()
    
    frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
    frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)

    backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
    forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))



    backward_frames = backward_frames.reshape(b,t-1,3,h,w)
    forward_frames = forward_frames.reshape(b,t-1,3,h,w)
    save_image(backward_frames[0,0], './debug_results/gt_backword_2.png')
    save_image(backward_frames[0,1], './debug_results/gt_backword_3.png')
    save_image(forward_frames[0,0], './debug_results/gt_forword_1.png')
    save_image(forward_frames[0,1], './debug_results/gt_forword_2.png')




def examine_stda_module():

    device = 'cuda:0'

    first_scale_encoder_second = torch.load('./debug_results/encoder_2nd.pt').to(device)
    flow_forward = torch.load('./debug_results/flow_forward.pt').to(device)
    flow_backward = torch.load('./debug_results/flow_backward.pt').to(device)


    mma = DeformableAttnBlock(n_heads=4,d_model=128,n_levels=3,n_points=12).to(device)
    msa = DeformableAttnBlock_FUSION(n_heads=4,d_model=128,n_levels=3,n_points=12).to(device)

    frame,srcframe = mma(first_scale_encoder_second,first_scale_encoder_second,flow_forward,flow_backward)
    first_scale_encoder_second_out = msa(frame,srcframe,flow_forward,flow_backward)

    print(first_scale_encoder_second_out.shape)



if __name__ == '__main__':
    examine_stda_module()

