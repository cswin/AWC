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

* pytoch
* albumentations

## Setup
* Check out the repo.<br>
`https://github.com/cswin/AWC.git`
* Download the processed OD and OC image ROIs from:<br>
[`https://drive.google.com/open?id=1Gi1lT7Sha1UAWMlGiPbU3S4kjd8yKWg1`](https://drive.google.com/open?id=1Gi1lT7Sha1UAWMlGiPbU3S4kjd8yKWg1)

### Usage
1. Train the model:
 
   ```shell
   cd src
   python main.py
   ```
2. Predict the masks:

   ```shell
   python predict.py
   ```
3. Evaluate the model:

   Coming soon.
   
### Questions

Further questions, please feel free to contact `pliu1 at ufl.edu` or `bkong at uncc.edu`
