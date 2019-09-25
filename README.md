# CFEA-Pytorch


## Pytorch implementation of CFEA.

This is a Pytorch implementation of the paper "CFEA: Collaborative Feature Ensembling Adaptation for Domain Adaptation in Unsupervised Optic Disc and Cup Segmentation". 

If you use this code in your research please consider citing

>@inproceedings{liumiccai2019,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; authors = {Peng Liu and Bin Kong and Zhongyu Li and Shaoting Zhang and Ruogu Fang},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          title = {CFEA: Collaborative Feature Ensembling Adaptation for Domain Adaptation in Unsupervised Optic Disc and Cup Segmentation},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          year = {2019}<br>
}
## Requirements

* python 3.6
* pytoch 1.0.0
* albumentations
 
### Usage
1. uncompress the sample data from data/

2. Train the model:
 
   ```shell
   cd src
   python train.py
   ```
3. Predict the masks:

   ```shell
   python predict.py
   ```

   
### Questions

Further questions, please feel free to contact `pliu1 at ufl.edu` or `bkong at uncc.edu`
