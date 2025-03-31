# Modified STDANet

Modified **Video Deblurring Model STDANet**, and official implementation of [**Edge-enhanced STDANet**](https://ieeexplore.ieee.org/document/10734118). This project is based on [STDAN](https://github.com/huicongzhang/STDAN).

## Requirements
#### 1.  Clone the Code Repository

```
git clone https://github.com/stmrym/STDAN_modified.git
```
#### 2.  Install Pytorch Denpendencies

```
conda create -n STDAN python=3.7 
conda activate STDAN
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

#### 3.  Install Python Denpendencies and Build PyTorch Extensions

```
cd STDAN_modified
sh install.sh
```

## Datasets

### 1. Download Datasets
We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release), and [BSD](https://github.com/zzh-tech/ESTRNN) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing)

### 2. Prepare JSON Files
Specify input data for the model using json files.

- #### BSD Dataset
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

- #### GoPro Dataset
Run [datasets/make_GOPRO_json.py](datasets/make_GOPRO_json.py) to create **GOPRO_train.json** and **GOPRO_valid.json** respectively.
```Python
# Path to GOPRO dataset
dataset_path = '../../dataset/GOPRO_Large'

# GOPRO dataset attribute 'train', or 'test'
phase = 'test'

# attributes to be assigned to JSON file
json_phase = 'valid'
```

- #### Your Own Dataset
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



## Run
To train, resume, and test , you can simply use the following command (e.g., using [config/sample_config.yml](config/sample_config.yml)):
```
python runner.py config/sample_config.yml
```


## License
This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STDAN](https://github.com/huicongzhang/STDAN).








