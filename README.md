# modified STDANet

Modified **video deblurring model STDANet.** This project is based on [STDAN](https://github.com/huicongzhang/STDAN).

## Changes from the original

- We save output images (**Output**) and visualized flow images (**Flow map**) can be saved.
- We added [valid.py](core/valid.py) and validation is available during training.
- You can train/valid/test on multiple datasets. (Please make json files and modify the config file)

### Output examples
|Input (blurred)|Output (deblurred)|
|:---:|:---:|
|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/e18abdd8-f481-4d48-9ac0-b3a8b5aadd70">|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/8a843d47-9944-4c0c-bb2d-3fe96bed8912">|
|**Flow vector**|**Flow angle**|
|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/9661e9fb-bc5b-406f-af1f-18cc414a4c6d">|<img width="100%" src="https://github.com/stmrym/STDANet_modified/assets/114562027/d4726cca-f5ee-499e-b45b-360ef6fd97fc">|

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


To train and test STDAN, you can simply use the following command (e.g., using config/config_1.py):

(Please replace [] with your path.)
```
python runner.py config/congig_1
```

In [config_1.py](config/config_1.py), there are more settings of testing and training. 



## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STDAN](https://github.com/huicongzhang/STDAN).








