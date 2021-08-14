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
import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils.utils import AverageMeter, str2bool

net = "12"
if net == "16":
    from network.SCnet16 import SpatialConfigurationNet
    from dataset.dataset16 import Dataset
else:
    from network.SCnet import SpatialConfigurationNet
    from dataset.datasettest import Dataset

from losses import LovaszLossSoftmax, LovaszLossHinge, dice_coeff

import cv2 
import math

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model_one',
                        help='model name')

    args = parser.parse_args()

    return args

def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output)
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)



def heat_map(data, h, w):
    """
    绘制散点热图
    Args:
        data: 数据结构为list[(int，int，value)]
        map_size: 画布大小
    Returns:热力图
    """
    map = np.array([0] * (h*w), dtype=np.uint8).reshape((h, w))
 
    for d in data:
        u = d[0]
        v = d[1]
        val = min(255, int(d[2] * 200))
        attention(u, v, val, map)
 
    return map


def attention(u, v, val, map, r=20):
    shape = map.shape
    w, h = shape[0], shape[1]
 
    intensity = np.linspace(val, 0, r, dtype=np.uint8)
 
    for x in range(max(0, u-r), min(w, u+r)):
        for y in range(max(0, v-r), min(h, v+r)):
            distance = math.ceil(math.sqrt(pow(x-u, 2) + pow(y-v, 2)))
 
            if distance < r:
                if map[x][y] == 0:
                    map[x][y] = intensity[distance]
                else:
                    map[x][y] = max(map[x][y], intensity[distance])

def main():
    args = parse_args()

    with open('../models/model_ver/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config['name'] = 'testdata'
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['model'])
    model = SpatialConfigurationNet(num_labels=config['num_classes'])

    model = model.cuda()

    # Data loading code
    img_names = os.listdir('../dataset/TestData/')


    model.load_state_dict(torch.load('../models/model_ver/model.pth'))
    model.eval()

    val_transform = None

    
    for c in range(len(img_names)):
        os.makedirs(os.path.join('../outputs', config['name'], img_names[c]), exist_ok=True)

    for i in range(len(img_names)):
        temp = sitk.ReadImage("../dataset/TestData/{}/MR_512.nii.gz".format(img_names[i]), sitk.sitkFloat32)
        temp = sitk.GetArrayFromImage(temp)
        print(len(temp))
        FinalMask = np.zeros((len(temp),256, 128))
        for j in range(len(temp) - 12 + 1):
            val_dataset = Dataset(img_names=[img_names[i]],
                    img_dir='../dataset/TestData/',
                    transform=val_transform,
                    num_classes = config['num_classes'],
                    train = False,
                    start = j)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                drop_last=False)
            with torch.no_grad():
                for input, center, meta in tqdm(val_loader, total=len(val_loader)):
        
                    input = input.cuda()
                    # compute output
                    output = model(input)[0]
                    output = torch.sigmoid(output).cpu().numpy()
                    
                    output1 =  np.where(output > 0.5, 1, 0)
                    FinalMask[j:j+12,:,:] += output1[0]
                FinalMask =  np.where(FinalMask > 0.5, 1, 0)
            torch.cuda.empty_cache()
        np.save('../outputs/{}/{}/PredVerAll.npy'.format(config['name'], meta['img_name'][0]), FinalMask)
        FinalImgO = np.zeros((256, 128))
        for k in range(len(temp)):
            FinalImgO += FinalMask[k,:,:]
        FinalImgO = (FinalImgO != 0)
        FinalImgO = (FinalImgO * 255).astype('uint8')
        cv2.imwrite(os.path.join('../outputs', config['name'], meta['img_name'][0],  'PredVer' + '.bmp'),
                        FinalImgO, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
if __name__ == '__main__':
    main()
