#!/usr/bin/python

from pickle import FALSE
from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.SEED                          = 1
__C.CONST.CONFIG_NAME                   = 'my_config_1'
__C.CONST.DEVICE                        = '0'                             # gpu_ids
__C.CONST.NUM_WORKER                    = 8                               # number of data workers
# __C.CONST.WEIGHTS                       = 'exp_log/train/F_2024-04-02T075334_ESTDAN_Stack_BSD_3ms24ms_GOPRO/checkpoints/latest-ckpt.pth.tar'
# __C.CONST.WEIGHTS                       = 'exp_log/train/2024-05-31T041732_F_STDAN_BSD_3ms24ms_GOPRO/checkpoints/latest-ckpt.pth.tar' # data weights path
__C.CONST.WEIGHTS                       = ''
__C.CONST.TRAIN_BATCH_SIZE              = 8 # original: 8
__C.CONST.EVAL_BATCH_SIZE               = 1 # original: 1
__C.CONST.PREFIX                        = 'F'  # This strings will be added to output_dir_path

#
# Dataset, logs and checkpoint Directories
#
__C.DATASET                             = edict()
__C.DIR                                 = edict()
__C.DATASET.TRAIN_DATASET_LIST          = ['BSD_3ms24ms', 'GOPRO']       
__C.DIR.TRAIN_IMAGE_BLUR_PATH_LIST      = [ 
                                            '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png',     # %s, %s, %s: phase, seq_name, image_name template
                                            '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'
                                            ]
__C.DIR.TRAIN_IMAGE_CLEAR_PATH_LIST     = [ 
                                            '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png',    # %s, %s, %s: phase, seq_name, image_name template
                                            '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
                                            ]
__C.DIR.TRAIN_JSON_FILE_PATH_LIST       = [ 
                                            '../STDAN_modified/datasets/BSD_3ms24ms_train_1000.json',
                                            '../STDAN_modified/datasets/GOPRO_train_1000.json'
                                            ]
# __C.DIR.TRAIN_JSON_FILE_PATH_LIST       = [ 
                                            # '../STDAN_modified/datasets/BSD_3ms24ms_train_debug.json',
                                            # '../STDAN_modified/datasets/GOPRO_train_debug.json'
                                            # ]

__C.DATASET.VAL_DATAET_LIST             = [
                                            'BSD_3ms24ms',
                                            'GOPRO'
                                            ]       
__C.DIR.VAL_IMAGE_BLUR_PATH_LIST        = [
                                             '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png',     # %s, %s, %s: phase, seq_name, image_name
                                            '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'    
                                            ]   
__C.DIR.VAL_IMAGE_CLEAR_PATH_LIST       = [ 
                                            '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png',    # %s, %s, %s: phase, seq_name, image_name
                                            '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
                                            ]
__C.DIR.VAL_JSON_FILE_PATH_LIST         = [ 
                                            '../STDAN_modified/datasets/BSD_3ms24ms_test.json',    
                                            '../STDAN_modified/datasets/GOPRO_valid.json'
                                            ]
# __C.DIR.VAL_JSON_FILE_PATH_LIST         = [ 
                                            # '../STDAN_modified/datasets/BSD_3ms24ms_valid_debug.json',
                                            # '../STDAN_modified/datasets/GOPRO_valid_debug.json'
                                            # ]

__C.DATASET.TEST_DATASET_LIST           = ['BSD_3ms24ms']       # Arbitary output name
__C.DIR.TEST_IMAGE_BLUR_PATH_LIST       = [
                                            '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png'              # %s, %s, %s: phase, seq_name, image_name
                                            ]   
__C.DIR.TEST_IMAGE_CLEAR_PATH_LIST      = [ '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png'
                                            ]
__C.DIR.TEST_JSON_FILE_PATH_LIST        = [ '../STDAN_modified/datasets/BSD_3ms24ms_test.json'
                                            ]

__C.DIR.OUT_PATH                        = './exp_log'         # logs path

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DEBLURNETARCH               = 'ESTDAN_v2'             
__C.NETWORK.PHASE                       = 'train'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.USE_STACK                   = True
__C.NETWORK.USE_ORTHO_EDGE              = True                    # use for only ESTDAN
__C.NETWORK.INPUT_LENGTH                = 5
__C.NETWORK.INPUT_CHANNEL               = 3
__C.NETWORK.NUM_FEAT                    = 32
__C.NETWORK.OUTPUT_CHANNEL              = 3
__C.NETWORK.NUM_RESBLOCK                = 3
__C.NETWORK.KERNEL_SIZE                 = 5
__C.NETWORK.SOBEL_OUT_CHANNEL           = 2                      # use for only ESTDAN



__C.LOSS                                = edict()
__C.LOSS_DICT_LIST                      = [ {'name': 'L1Loss',          'func': 'l1Loss',           'weight': 1},
                                            {'name': 'WarpMSELoss',     'func': 'warp_loss',        'weight': 0.05},
                                            {'name': 'FFTLoss',         'func': 'FFTLoss',          'weight': 0.05}
                                            # {'name': 'MotionEdgeLoss',  'func': 'motion_edge_loss', 'weight': 0.05},
                                            # {'name': 'OrthogonalEdgeLoss',  'func': 'orthogonal_edge_loss', 'weight': 0.05}
                                            ]
__C.LOSS.MOTION_REQUIRES_GRAD        = True                    # If False, fix weights for motion estimator
__C.LOSS.SOBEL_REQUIRES_GRAD         = True
#
# RAFT options
#
__C.RAFT                                = edict()
__C.RAFT.CONFIG_FILE                    = '../STDAN_modified/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py'
__C.RAFT.CHECKPOINT                     = '../STDAN_modified/mmflow/checkpoints/raft_8x2_100k_mixed_368x768.pth'

#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue


#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = False
__C.TRAIN.NUM_EPOCHES                   = 1201   # original: 1200                   # maximum number of epoches
__C.TRAIN.LEARNING_RATE                 = 1e-4 # original: 1e-4
__C.TRAIN.LR_MILESTONES                 = [400,600,800,1000]   
# __C.TRAIN.LR_MILESTONES                 = [2000]   
__C.TRAIN.LR_DECAY                      = 0.5                   # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                    # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                    # regularization of weight, default: 0
__C.TRAIN.SAVE_FREQ                     = 50                     # weights will be overwritten every save_freq epoch

#
# eval options
#
__C.EVAL                                 = edict()
__C.EVAL.VALID_FREQ                      = 50
__C.EVAL.VISUALIZE_FREQ                  = 50                    # frequency of vilidation visualization
__C.EVAL.SAVE_FLOW                       = True
__C.EVAL.USE_TENSORBOARD                 = True
__C.EVAL.CALC_METRICS                    = True