#!/usr/bin/python

from pickle import FALSE
from easydict import EasyDict as edict
import os
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = '0'                             # gpu_ids
__C.CONST.NUM_WORKER                    = 8                               # number of data workers
# __C.CONST.WEIGHTS                       = 'exp_log/train/2023-11-17T084827_STDAN_Stack_BSD_3ms24ms/checkpoints/latest-ckpt.pth.tar'
__C.CONST.WEIGHTS                       = 'exp_log/train/2023-11-22T163558_STDAN_Stack_BSD_3ms24ms/checkpoints/ckpt-epoch-0905.pth.tar' # data weights path
# __C.CONST.WEIGHTS                       = ''
__C.CONST.TRAIN_BATCH_SIZE              = 4 # original: 8
__C.CONST.VAL_BATCH_SIZE                = 1
__C.CONST.TEST_BATCH_SIZE               = 1 # original: 1
# __C.CONST.DEBUG_PREFIX                  = '20231129_'  # This strings will be added to output_dir_path
__C.CONST.DEBUG_PREFIX                  = 'debug_'  # This strings will be added to output_dir_path

#
# Dataset, logs and checkpoint Directories
#
__C.DATASET                             = edict()
__C.DIR                                 = edict()
__C.DATASET.TRAIN_DATASET_NAME          = 'GOPRO'       # available options:  'DVD','GOPRO','BSD_1ms8ms','BSD_2ms16ms','BSD_3ms24ms'
__C.DIR.TRAIN_IMAGE_BLUR_PATH           = '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'   # %s, %s, %s: phase, seq_name, image_name
__C.DIR.TRAIN_IMAGE_CLEAR_PATH          = '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
__C.DIR.TRAIN_JSON_FILE_PATH            = '../STDAN_modified/datasets/GOPRO_train.json'

__C.DATASET.VAL_DATAET_LIST             = ['BSD_3ms24ms', 'GOPRO']
__C.DIR.VAL_IMAGE_BLUR_PATH_LIST        = [
                                                '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png',
                                                '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'
                                            ]   # %s, %s, %s: phase, seq_name, image_name
__C.DIR.VAL_IMAGE_CLEAR_PATH_LIST       = [
                                                '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png',
                                                '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
                                            ]
__C.DIR.VAL_JSON_FILE_PATH_LIST         = [
                                                '../STDAN_modified/datasets/BSD_3ms24ms_valid.json',
                                                '../STDAN_modified/datasets/GOPRO_valid.json'
                                            ]

__C.DATASET.TEST_DATASET_LIST           = ['ADAS']       # Arbitary output name
__C.DIR.TEST_IMAGE_BLUR_PATH_LIST       = [
                                                '../dataset/ADAS/%s/target/%s/%s.jpg',
                                            ]   # %s, %s, %s: phase, seq_name, image_name
__C.DIR.TEST_IMAGE_CLEAR_PATH_LIST      = [
                                                '../dataset/ADAS/%s/target/%s/%s.jpg'
                                            ]
__C.DIR.TEST_JSON_FILE_PATH_LIST        = [
                                                '../STDAN_modified/datasets/ADAS.json',
                                            ]

__C.DIR.OUT_PATH                        = './exp_log'         # logs path

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DEBLURNETARCH               = 'STDAN_Stack'             
__C.NETWORK.PHASE                       = 'train'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.TAG                         = __C.DATASET.TRAIN_DATASET_NAME     # logs folder tag


#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue
__C.DATA.INPUT_LENGTH                   = 5

#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = False
__C.TRAIN.NUM_EPOCHES                   = 1000   # original: 1200                   # maximum number of epoches
__C.TRAIN.LEARNING_RATE                 = 1e-4 # original: 1e-4
# __C.TRAIN.LR_MILESTONES                 = [400,600,800,1000]   
__C.TRAIN.LR_MILESTONES                 = [300,400,500,600,700]   
__C.TRAIN.LR_DECAY                      = 0.5                   # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                    # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                    # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 100                    # print step
__C.TRAIN.SAVE_FREQ                     = 5                     # weights will be overwritten every save_freq epoch

#
# Val options
#
__C.VAL                                 = edict()
__C.VAL.VISUALIZE_FREQ                  = 100                    # frequency of vilidation visualization

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.PRINT_FREQ                     = 5
