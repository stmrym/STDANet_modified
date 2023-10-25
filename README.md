# modified STDAN

This project is based on [STDAN](https://github.com/huicongzhang/STDAN)

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


To train STDAN, you can simply use the following command:

```
bash run_code_train.sh
```

To test STDAN, you can simply use the following command:

```
bash run_code_test.sh
```

In [here](config.py), there are more settings of testing and training. 


## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STDAN](https://github.com/huicongzhang/STDAN)








