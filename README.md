# D2-Net: A Trainable CNN for Joint Detection and Description of Local Features
[Paper on arXiv](https://arxiv.org/abs/1905.03561), [Project page](https://dsmn.ml/publications/d2-net.html)
    
## Getting started

Python 3.6+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
```

## Downloading the models

```bash
mkdir models
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
```

## Data Preparation
因为数据集不统一，所以目前要运行只能这样

The layout should look like this 

```
code
|-- test_data
|    ## this part is dataset
|    |-- 1100a
|    |    |--Img
|    |    |    |-- 1.img
|    |    |    |-- 2.img
|    |    |    ....
|    |-- 1044
|    |-- 733
|    |-- 33
|
|    ## this part is what you need to prepare
|    ## we supply these .txt, copy them into "test_data"!!!
|    |-- img_1100a.txt
|    |-- img_1044.txt
|    |-- img_733.txt
|    |-- img_33.txt
|    |-- 1100a_1044.txt
|    |-- 1044_733.txt
|    |-- 733_33.txt
|
|    ## the followings are sequecses
|    |-- matches_1044_733
|    |-- 1044_733_result.txt
```

## Test start
you can use two script to get what for in "Data Preparation"  
maybe some code needed to be changed 
```
./getpath.sh > img_1044.txt
./getpath.sh > img_733.txt
python combine.py

## you also need compute.py
```

## Test Run
If you want this run in your network and dataset, After you get the keypoints and descriptor, use the "Tracker0" to match, use the compute.py to compute error. But you must make sure "img_list_path.txt" looks like ours
```
python d2_match.py --data_path /home/chenxiang/code/d2-net/test
                   --pair 1044_733
```