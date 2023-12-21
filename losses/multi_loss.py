import cv2
import torch
import torch.nn as nn
from utils.network_utils import *
from models.submodules import warp

def mseLoss(output, target):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(output, target)

    return MSE

def warp_loss(output_dict:dict, gt_seq:torch.tensor):
    
    b,t,c,h,w = gt_seq.shape
    down_simple_gt = F.interpolate(gt_seq.reshape(-1,c,h,w), size=(h//4, w//4),mode='bilinear', align_corners=True).reshape(b,t,c,h//4,w//4)
    frames_list = down_simple_gt
    
    n, t, c, h, w = frames_list.size()
    flow_forwards = output_dict['flow_fowards']
    flow_backwards = output_dict['flow_backwards']

    forward_loss = 0
    backward_loss = 0
    forward_mse_loss = nn.L1Loss()
    backward_mse_loss = nn.L1Loss()
    for idx in [[0,1,2],[1,2,3],[2,3,4],[1,2,3]]:
        frames = frames_list[:,idx,:,:,:]
        for flow_forward,flow_backward in zip(flow_forwards,flow_backwards):
            frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            backward_frames = warp(frames_1,flow_backward.reshape(-1, 2, h, w))
            forward_frames = warp(frames_2,flow_forward.reshape(-1, 2, h, w))
            forward_loss += forward_mse_loss(forward_frames,frames_1)
            backward_loss += backward_mse_loss(backward_frames,frames_2)
    return (0.5*forward_loss + 0.5*backward_loss)/len(flow_forwards)


def l1Loss(output_dict:dict, gt_seq:torch.tensor):

    output_imgs = torch.cat([output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']],dim=1)
    t_gt_seq = torch.cat([gt_seq[:,1,:,:,:],gt_seq[:,2,:,:,:],gt_seq[:,3,:,:,:],gt_seq[:,2,:,:,:]],dim=1)

    l1_loss = nn.L1Loss()
    l1 = l1_loss(output_imgs, t_gt_seq)
    return l1

def PSNR(output, target, max_val = 1.0,shave = 4):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target[:,:,shave:-shave,shave:-shave] - output[:,:,shave:-shave,shave:-shave], 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)

def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

def tensor_image_canny(save_name, tensor_image):
    tensor_image = tensor_image[0].permute(1,2,0).cpu().detach()*255

    np_image = tensor_image.numpy()
    gray_image = cv2.cvtColor(np.clip(np_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    med_val = np.median(gray_image)
    sigma = 0.11  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    np_edge = cv2.Canny(gray_image, threshold1=min_val, threshold2=max_val)
    
    cv2.imwrite(f'{save_name}_edge.png', np_edge)



def save_image(save_name, out_tensor):
        output_image = out_tensor.cpu().detach()*255
        output_image = output_image[0].permute(1,2,0)


        output_image = output_image.numpy().copy()
        output_image_bgr = cv2.cvtColor(np.clip(output_image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f'{save_name}.png', output_image_bgr)

def motion_edge_loss(output_dict:dict, gt_seq:torch.tensor):
    
    recons_1, recons_2, recons_3, out = output_dict['recons_1'], output_dict['recons_2'], output_dict['recons_3'], output_dict['out']
    flow_fowards, flow_backwards = output_dict['flow_fowards'], output_dict['flow_backwards']

    tensor_image_canny('recons_1', recons_1)
    save_image('recons_2', recons_2)
    save_image('recons_3', recons_3)
    save_image('rout', out)


def calc_update_losses(output_dict:dict, gt_seq:torch.tensor, losses_dict_list:list, total_losses, batch_size:int):

    total_loss = 0
    for losses_dict in losses_dict_list:
        loss = eval(losses_dict['func'])(output_dict, gt_seq) * losses_dict['weight']   # Calculate loss
        losses_dict['avg_meter'].update(loss.item(), batch_size)    # Update loss
        total_loss += loss

    total_losses.update(total_loss.item(), batch_size) # Update total losses
    return total_loss, total_losses, losses_dict_list