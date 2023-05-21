# Super Resolution 

# Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Training](#training)
    <!-- - [Adding to Chrome](#adding-to-chrome) --> 
- [Test](#test)  
- [Results](#results) 

# Introduction

This project implements simpler and faster End-to-End, Lightweight SISR methods under 4× up-scale, summarizes and compares performance of them. The traditional models include Nearest Neighbor, Biliear and Bicubic. The deep learning-based models include SRCNN, FSRCNN, MAFFSRN,  NEAREST, BSRN, DRRN, ESPCN, RFDN and IMDN

# Installation 

``` shell
conda create -n sr python=3.8 -y

conda activate sr

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install --upgrade pip

``` 

Run the requirements.txt file to install all the dependencies.

``` shell
pip install -r requirements.txt
```  

# Training 
Taking the IMDN model as an example, the training command is as follows:

``` shell
python train.py --model IMDN
```  

# Test 
Taking the IMDN model as an example, the test command is as follows:

``` shell
python test.py --model IMDN
``` 
The PSNR and SSIM of the test results will be provided.  

# Results
The qulitative results of the test are shown below.

![PSNR & SSIM](/png/PSNR.png) 

Samples generated by models are also shown in the 'output' folders. There are 10 samples for each model. 

![Samples](/png/samples.png)








