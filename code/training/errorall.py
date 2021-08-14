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
from dataset.dataset import zxt, Dataset2
from losses import LovaszLossSoftmax, LovaszLossHinge, dice_coeff




img_names = os.listdir('../outputs/data/')[:-3]
    
Ex = []
Ey = []
hashtable = np.load('../outputs/data/LocPos.npy', allow_pickle=True).item()
img_names.remove("Case39")
img_names.remove("Case12")
for i in range(len(img_names)):
    path = "../outputs/data/" + img_names[i] + '/Predfinal.csv'
    path2 = "../outputs/data/" + img_names[i] + '/RealPoints.csv'
    pred = pd.read_csv(path)
    real = pd.read_csv(path2)
    
    for j in range(19):
        errX = pred['x'][j] - real['x'][j]
        errY = pred['y'][j] - real['y'][j]
        if not np.isnan(errX) and not np.isnan(errY):
            if abs(errX) > 5:
                print("X")
                print(j)
                print(img_names[i])
            if abs(errY) > 5:
                print("Y")
                print(j)
                print(img_names[i])
            Ex.append(abs(errX))
            Ey.append(abs(errY))
            
          