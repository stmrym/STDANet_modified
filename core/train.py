import os
from turtle import backward
from utils import log, util
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from losses.multi_loss import *
from time import time

from core.valid import valid
# from models.VGG19 import VGG19
# from utils.network_utils import flow2rgb
from tqdm import tqdm



def train(cfg, init_epoch, 
        train_transforms, val_transforms,
        deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
        ckpt_dir, visualize_dir, 
        tb_writer, Best_Img_PSNR, Best_Epoch):

    n_itr = 0
    # Training loop
    torch.manual_seed(0)

    # Set up data loader
    dataset_list = []
    for train_dataset_name, train_image_blur_path, train_image_clear_path, train_json_file_path\
        in zip(cfg.DATASET.TRAIN_DATASET_LIST, cfg.DIR.TRAIN_IMAGE_BLUR_PATH_LIST, cfg.DIR.TRAIN_IMAGE_CLEAR_PATH_LIST, cfg.DIR.TRAIN_JSON_FILE_PATH_LIST):
            
            # Load each dataset and append to list
            dataset_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(
                image_blur_path = train_image_blur_path, 
                image_clear_path = train_image_clear_path,
                json_file_path = train_json_file_path,
                input_length = cfg.DATA.INPUT_LENGTH)
            
            dataset = dataset_loader.get_dataset(transforms = train_transforms)
            dataset_list.append(dataset)

            case_num = int(len(dataset)) * cfg.CONST.TRAIN_BATCH_SIZE
            log.info(f'[TRAIN] Dataset [{train_dataset_name}] loaded. Train case: {case_num}')

    # Concat all dataset
    all_dataset = torch.utils.data.ConcatDataset(dataset_list)

    total_case_num = int(len(all_dataset)) * cfg.CONST.TRAIN_BATCH_SIZE
    log.info(f'[TARIN] Total train case: {total_case_num}')
    assert total_case_num != 0, f'[TRAIN] Total train case empty!'

    # Creating dataloader
    train_data_loader = torch.utils.data.DataLoader(
        dataset=all_dataset,
        batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

    # Start epoch
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):

        # Batch average meterics
        losses_dict_list = []
        for loss_config_dict in cfg.LOSS_DICT_LIST:
            losses_dict = loss_config_dict.copy()
            losses_dict['avg_meter'] = utils.network_utils.AverageMeter()
            losses_dict_list.append(losses_dict)

        total_losses = utils.network_utils.AverageMeter()

        img_PSNRs_mid = utils.network_utils.AverageMeter()
        img_PSNRs_out    = utils.network_utils.AverageMeter()

        tqdm_train = tqdm(train_data_loader)
        tqdm_train.set_description(f'[TRAIN] [Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}]')

        for seq_idx, (name, seq_blur, seq_clear) in enumerate(tqdm_train):
            # Get data from data loader
            seq_blur  = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_blur]
            seq_clear = [utils.network_utils.var_or_cuda(img).unsqueeze(1) for img in seq_clear]
            
            # switch models to training mode
            deblurnet.train()

            # Train the model
            input_seq = torch.cat(seq_blur,1)
            gt_seq = torch.cat(seq_clear,1)
            
            
            # recons_1, recons_2, recons_3, out, flow_forward, flow_backward = deblurnet(input_seq)
            output_dict = deblurnet(input_seq) # {'recons_1': first output, 'recons_2': second output, 'recons_3': third output, 'out': final output, 'flow_forwards': fowards_list, 'flow_backwards': backwards_list}
            
            # Calculate & update loss
            total_loss, total_losses, losses_dict_list = calc_update_losses(output_dict=output_dict, gt_seq=gt_seq, losses_dict_list=losses_dict_list, total_losses=total_losses, batch_size=cfg.CONST.TRAIN_BATCH_SIZE)

            img_PSNR_out = util.calc_psnr(output_dict['out'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_out.update(img_PSNR_out, cfg.CONST.TRAIN_BATCH_SIZE)

            img_PSNR_mid = util.calc_psnr(output_dict['recons_2'].detach(),gt_seq[:,2,:,:,:].detach())
            img_PSNRs_mid.update(img_PSNR_mid, cfg.CONST.TRAIN_BATCH_SIZE)

            deblurnet_solver.zero_grad()
            total_loss.backward()
            deblurnet_solver.step()
            
            n_itr = n_itr + 1

            # Tick / tock
            tqdm_train.set_postfix_str(f'  TotalLoss {total_losses} PSNR_mid {img_PSNRs_mid} PSNR_out {img_PSNRs_out}')        
            
        # Append epoch loss to TensorBoard
        for losses_dict in losses_dict_list:
            tb_writer.add_scalar(f'Loss_TRAIN/{losses_dict["name"]}', losses_dict["avg_meter"].avg, epoch_idx)
        
        tb_writer.add_scalar('Loss_TRAIN/TotalLoss', total_losses.avg, epoch_idx)
        tb_writer.add_scalar('PSNR/TRAIN', img_PSNRs_out.avg, epoch_idx)
        tb_writer.add_scalar('lr/lr', deblurnet_lr_scheduler.get_last_lr()[0], epoch_idx)
        deblurnet_lr_scheduler.step()

        log.info(f'[TRAIN][Epoch {epoch_idx}/{cfg.TRAIN.NUM_EPOCHES}] PSNR_mid: {img_PSNRs_mid.avg}, PSNR_out: {img_PSNRs_out.avg}, PSNR_best: {Best_Img_PSNR}')

        if epoch_idx % cfg.VAL.VALID_FREQ == 0:
            
            # Validation for each dataset list
            for val_dataset_name, val_image_blur_path, val_image_clear_path, val_json_file_path\
                in zip(cfg.DATASET.VAL_DATAET_LIST, cfg.DIR.VAL_IMAGE_BLUR_PATH_LIST, cfg.DIR.VAL_IMAGE_CLEAR_PATH_LIST, cfg.DIR.VAL_JSON_FILE_PATH_LIST):
                val_loader = utils.data_loaders.VideoDeblurDataLoader_No_Slipt(
                    image_blur_path = val_image_blur_path, 
                    image_clear_path = val_image_clear_path,
                    json_file_path = val_json_file_path,
                    input_length = cfg.DATA.INPUT_LENGTH)

                save_dir = os.path.join(visualize_dir, val_dataset_name, 'epoch-' + str(epoch_idx).zfill(4))

                # Validation
                Best_Img_PSNR, Best_Epoch = valid(
                    cfg = cfg,
                    val_dataset_name = val_dataset_name,
                    epoch_idx = epoch_idx, init_epoch = init_epoch,
                    Best_Img_PSNR = Best_Img_PSNR,
                    Best_Epoch = Best_Epoch, 
                    ckpt_dir = ckpt_dir,
                    save_dir = save_dir,
                    val_loader = val_loader, 
                    val_transforms = val_transforms,
                    deblurnet = deblurnet,
                    tb_writer = tb_writer)
                
        
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx)), \
                                                    epoch_idx, deblurnet,deblurnet_solver, \
                                                    Best_Img_PSNR, Best_Epoch)
            
        
        utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'latest-ckpt.pth.tar'), \
                                                    epoch_idx, deblurnet, deblurnet_solver,\
                                                    Best_Img_PSNR, Best_Epoch)
        
    # Close SummaryWriter for TensorBoard
    tb_writer.close()