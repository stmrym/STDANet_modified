# modified STDAN

This project is based on [STDAN](https://github.com/huicongzhang/STDAN)

## Changes from the original

- We save output images and visualized flow images at the time of testing.
- We added [valid.py](core/valid.py) and split the BSD dataset into train, valid and test by [make_BSD_json.py](datasets/make_BSD_json.py).

### Output examples
|Input|Deblurred output|Flow vector|Flow angle|
|---|---|---|---|
|<img width="200" src="https://github.com/stmrym/STDANet_modified/assets/114562027/44221eaa-4256-4493-808e-6ef4193e5fc9">|<img width="200" src="https://github.com/stmrym/STDANet_modified/assets/114562027/8a843d47-9944-4c0c-bb2d-3fe96bed8912">|<img width="230" src="https://github.com/stmrym/STDANet_modified/assets/114562027/9661e9fb-bc5b-406f-af1f-18cc414a4c6d">|<img width="220" src="https://github.com/stmrym/STDANet_modified/assets/114562027/d4726cca-f5ee-499e-b45b-360ef6fd97fc">|

## Datasets

We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release), [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) and [BSD](https://github.com/zzh-tech/ESTRNN) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- [DVD](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip)
- [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing)


## Prerequisites
#### Clone the Code Repository

```
git clone https://github.com/stmrym/STDAN_modified.git
```
### Install Pytorch Denpendencies

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

## Get Started


To train STDAN, you can simply use the following command (e.g., BSD_3ms24ms):

(Please replace [] with your path.)
```
python runner.py \
    --data_path=/[your dataset path]/BSD_3ms24ms \
    --json_path=./datasets/BSD_3ms24ms_train_val_test.json \
    --data_name=BSD_3ms24ms \
    --phase=train   
```

To test STDAN, you can simply use the following command (e.g., BSD3ms24ms):

(Please replace [] with your path.)
```
python runner.py \
    --data_path=/[your dataset path]/BSD_3ms24ms \
    --json_path=./datasets/BSD_3ms24ms_train_val_test.json \
    --data_name=BSD_3ms24ms \
    --phase=test \
    --weights=./exp_log/train/[yyyy-mm-ddThhmmss_STDAN_Stack_BSD_3ms24m]/checkpoints/[saved ckpt].pth.tar  
```

In [config.py](config.py), there are more settings of testing and training. 



## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STDAN](https://github.com/huicongzhang/STDAN)








