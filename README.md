# modified STDANet

Modified **video deblurring model STDANet.** This project is based on [STDAN](https://github.com/huicongzhang/STDAN).

## Changes from the original

- We save output images (**Output**) and visualized flow images (**Flow map**) can be saved.
- We added [valid.py](core/valid.py) and **validation** is available during training.
- You can train/valid/test on **multiple datasets**. (Please make json files and modify [config/config 1.py](config/config_1.py))
- You can add **multiple loss fuctions** by editing [config/config 1.py](config/config_1.py).

### Output examples
|Input (blurred)|Output (deblurred)|Flow map|
|:---:|:---:|:---:|
|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/ed56addf-e03a-4e8e-a5f7-6e2638e83a78">|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/4324dad1-7389-4997-8711-f27ec1eb0f90">|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/797f8fe4-408d-48fa-9836-b0ed2b15f015">


## Requirements
#### Clone the Code Repository

```
git clone https://github.com/stmrym/STDAN_modified.git
```
#### Install Pytorch Denpendencies

```
conda create -n STDAN python=3.7 
conda activate STDAN
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

#### Install Python Denpendencies and Build PyTorch Extensions

```
cd STDAN_modified
sh install.sh
```

## Datasets

### Download datasets
We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release), and [BSD](https://github.com/zzh-tech/ESTRNN) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing)

### Prepare json files
Specify input data for the model using json files.

- #### BSD dataset
Run [datasets/make_BSD_json.py](datasets/make_BSD_json.py) to create **BSD_train.json**, **BSD_valid.json**, and **BSD_test.json** respectively.
```Python
# Path to BSD dataset
dataset_path = '../../dataset/'

# BSD_1ms8ms, BSD_2ms16ms, or BSD_3ms24ms 
bsd_type = 'BSD_3ms24ms'

# BSD dataset attributes 'train', 'valid', or 'test'
phase = 'train'

# attributes to be assigned to JSON file
json_phase = 'train'
```

- #### GoPro dataset
Run [datasets/make_GOPRO_json.py](datasets/make_GOPRO_json.py) to create **GOPRO_train.json** and **GOPRO_valid.json** respectively.
```Python
# Path to GOPRO dataset
dataset_path = '../../dataset/GOPRO_Large'

# GOPRO dataset attribute 'train', or 'test'
phase = 'test'

# attributes to be assigned to JSON file
json_phase = 'valid'
```

- #### Your own dataset
Run [datasets/make_original_json_sample.py](datasets/make_original_json_sample.py). The dataset must consist of [PHASE], [VIDEO_SEQS], and [FRAME].

(e.g., ```~/dataset/my_dataset/test/001/000001.png```)
```Python
# dataset structure: [PATH_TO_DATASET]/[PHASE]/[VIDEO_SEQS]/[FRAME]
# Path to dataset 
dataset_path = '~/dataset/my_dataset'

# dataset attribute 'train', 'valid', or 'test'
phase = 'test'

# saved JSON file name
savename = 'my_dataset'
```



## Get Started
To train and test STDAN, you can simply use the following command (e.g., using [config/config_1.py](config/config_1.py)):
```
python runner.py config/congig_1
```

## Configuration
Using [config/config_1.py](config/config_1.py) as an example.

#### Common

```Python
# Arbitary config file name (used for print log)
__C.CONST.CONFIG_NAME                   = 'my_config_1'
# GPU ids
__C.CONST.DEVICE                        = '0'
# Nunber of data workers
__C.CONST.NUM_WORKER                    = 8
# If you 'resume' or 'test', set your data weights path. If you 'train' from the beginning, set ''.
__C.CONST.WEIGHTS                       = 'exp_log/train/yyyy-MM-ddTHHmmss_STDAN_Stack_BSD_3ms24ms_GOPRO/checkpoints/ckpt-epoch-0xxx.pth.tar'
# Training batch size
__C.CONST.TRAIN_BATCH_SIZE              = 4
# Validation batch size
__C.CONST.VAL_BATCH_SIZE                = 1
# Test batch size
__C.CONST.TEST_BATCH_SIZE               = 1
# When 'train', this strings will be added to output directory (e.g., exp_log/train/exp1_yyyy-MM-ddTHHmmss_STDAN_Stack_BSD_3ms24ms_GOPRO)
__C.CONST.PREFIX                        = 'exp1_'
```

#### Dataset, Directories
```Python
# Arbitary train dataset name in list format
__C.DATASET.TRAIN_DATASET_LIST          = ['BSD_3ms24ms', 'GOPRO']
# Blurred (input) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
# (e.g., phase='test', seq_name='000', image_name='000000')
__C.DIR.TRAIN_IMAGE_BLUR_PATH_LIST      = [ '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png',
                                            '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'
                                            ]
# Clear (GT) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
__C.DIR.TRAIN_IMAGE_CLEAR_PATH_LIST     = [ '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png',    # %s, %s, %s: phase, seq_name, image_name template
                                            '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
                                            ]
# Set the corresponding json files.
__C.DIR.TRAIN_JSON_FILE_PATH_LIST       = [ './datasets/BSD_3ms24ms_train.json',
                                            './datasets/GOPRO_train.json'
                                            ]

# Arbitary validation dataset name in list format
__C.DATASET.VAL_DATAET_LIST             = ['BSD_3ms24ms', 'GOPRO']
# Blurred (input) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
# (e.g., phase='test', seq_name='000', image_name='000000')
__C.DIR.VAL_IMAGE_BLUR_PATH_LIST        = [ '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png',     # %s, %s, %s: phase, seq_name, image_name
                                            '../dataset/GOPRO_Large/%s/%s/blur_gamma/%s.png'    
                                            ]
# Clear (GT) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
__C.DIR.VAL_IMAGE_CLEAR_PATH_LIST       = [ '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png',    # %s, %s, %s: phase, seq_name, image_name
                                            '../dataset/GOPRO_Large/%s/%s/sharp/%s.png'
                                            ]
# Set the corresponding json files.
__C.DIR.VAL_JSON_FILE_PATH_LIST         = [ './datasets/BSD_3ms24ms_valid.json',    
                                            './datasets/GOPRO_valid.json'
                                            ]

# Arbitary test dataset name in list format
__C.DATASET.TEST_DATASET_LIST           = ['BSD_3ms24ms']
# Blurred (input) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
__C.DIR.TEST_IMAGE_BLUR_PATH_LIST       = [
                                            '../dataset/BSD_3ms24ms/%s/%s/Blur/RGB/%s.png'              # %s, %s, %s: phase, seq_name, image_name
                                            ]
# Clear (GT) image path list for dataset list. Replace phase, seq_name, and image_name templates with %s
__C.DIR.TEST_IMAGE_CLEAR_PATH_LIST      = [ '../dataset/BSD_3ms24ms/%s/%s/Sharp/RGB/%s.png'
                                            ]
# Set the corresponding json files.
__C.DIR.TEST_JSON_FILE_PATH_LIST        = [ './datasets/BSD_3ms24ms_test.json'
                                            ]

# Output path of experiment
__C.DIR.OUT_PATH                        = './exp_log'
```

#### Network
```'name'``` : Loss name, using for Tensorboard.

```'func'``` : Function name in [losses/multi_loss.py](losses/multi_loss.py).

```'weight'``` : Wight coefficient of the loss.
```Python
# Set the deblurring network ('STDAN_Stack' or 'STDAN_RAFT_Stack')
__C.NETWORK.DEBLURNETARCH               = 'STDAN_Stack'
# Network phase ('train', 'resume', or 'test')  
__C.NETWORK.PHASE                       = 'train'
# If False, fix weights for motion estimator
__C.NETWORK.MOTION_REQUIRES_GRAD        = True

# Set various losses in list format. Each loss is described in dictionary form.
__C.LOSS_DICT_LIST                      = [ {'name': 'L1Loss',          'func': 'l1Loss',           'weight': 1},
                                            {'name': 'WarpMSELoss',     'func': 'warp_loss',        'weight': 0.05},
                                            {'name': 'MotionEdgeLoss',  'func': 'motion_edge_loss', 'weight': 0.05}
                                            ]

# When using RAFT flow estimation, set the config and checkpoints.
__C.RAFT.CONFIG_FILE                    = './mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py'
__C.RAFT.CHECKPOINT                     = './mmflow/checkpoints/raft_8x2_100k_mixed_368x768.pth'
```

#### Validation
```Python
# Frequency of validation
__C.VAL.VALID_FREQ                      = 10
# Frequency of visualization of validation results
__C.VAL.VISUALIZE_FREQ                  = 50
# If True, saving flow map
__C.VAL.SAVE_FLOW                       = True
```

## License
This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STDAN](https://github.com/huicongzhang/STDAN).








