# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:35:39 2021

@author: Madmax
"""

import sys
sys.path.append("..")
import argparse
from glob import glob
import os
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import autocast 
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import cv2
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import SimpleITK as sitk
import matplotlib.pyplot as plt

from utils.utils import AverageMeter, str2bool
from network.SCnet import SpatialConfigurationNet
from network.archs import AttentionUNet, NestedUNet, ConvFour, R2UNet
from dataset.datasetnew import Dataset, Dataset2
from losses import LovaszLossSoftmax, LovaszLossHinge, dice_coeff




img_names = os.listdir('../outputs/data/')[:-3]
mask_dir = '../dataset/RawData/'
val_dir = '../outputs/data/'
w = 16
h = 24
#img_names =  ["Case12"]
for i in range(len(img_names)):
    print(i)
    file_path = mask_dir + img_names[i] + '/Mask.nii.gz'
    mask = sitk.ReadImage(file_path, sitk.sitkFloat32)
    mask = sitk.GetArrayFromImage(mask)
    mask_out = np.zeros((256,256))
    d = []
    for p in range(1, 20):
        mask_temp = mask == p
        mask_new = np.zeros((256,256))
        for j in range(len(mask_temp)):
            mask_new += mask_temp[j]
        mask_new = mask_new != 0
        xmean = 0
        ymean = 0
        for x in range(len(mask_new)):
            for y in range(len(mask_new[0])):
                if mask_new[x,y] != 0:
                    xmean += x
                    ymean += y
        xmean /= mask_new.sum()
        ymean /= mask_new.sum()
        d.append([xmean, ymean])
#        if p > 10:
#            mask_out += mask_new
#    mask_out = mask_out != 0
#    mask_out = (mask_out * 255).astype('uint8')
#    landmarks = pd.read_csv('../outputs/data/{}/Fianl.csv'.format(img_names[i]))
#    for k in range(len(d)):
#        if not np.isnan(d[k][1]) and not np.isnan(d[k][0]):
#            cv2.rectangle(mask_out, (max(0, int(d[k][1])-h), max(0, int(d[k][0])-w)), (min(256,int(d[k][1])+h), min(256, int(d[k][0])+w)), 255, 2)    
#    cv2.imwrite("../outputs/data/{}/FianlMask4.bmp".format(img_names[i]), mask_out)
#    for k in range(len(landmarks)):
#        cv2.rectangle(mask_out, (max(0, int(landmarks['y'][k]-h)), max(0, int(landmarks['x'][k]-w))), (min(256,int(landmarks['y'][k]+h)), min(256, int(landmarks['x'][k]+w))), 255, 2)
    save = pd.DataFrame(d, columns = ['x', 'y'])
#    cv2.imwrite("../outputs/data/{}/FianlMask3.bmp".format(img_names[i]), mask_out)
    save.to_csv("../outputs/data/{}/RealPoints.csv".format(img_names[i]),index=True,header=True)