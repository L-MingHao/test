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

from utils.utils import AverageMeter, str2bool

net = "12"
if net == "16":
    from network.SCnet16 import SpatialConfigurationNet
    from dataset.dataset16 import Dataset
else:
    from network.SCnet import SpatialConfigurationNet
    from dataset.datasetnew import Dataset

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

    with open('../models/model_ivds/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config['name'] = 'data'
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
    img_names = os.listdir('../dataset/RawData/')

    _, val_img_ids = train_test_split(img_names, test_size=0.99, random_state=40)

    model.load_state_dict(torch.load('../models/model_ivds/model.pth'))
    model.eval()

    val_transform = None

    val_dataset = Dataset(img_names=img_names,
                    img_dir='../dataset/RawData/',
                    transform=val_transform,
                    num_classes = config['num_classes'],
                    train = False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False)

    avg_meter = AverageMeter()
    
    hashtable = dict()
    
    hashtable2 = np.load('../outputs/data/LocPos.npy', allow_pickle=True).item()
    
    for c in range(len(img_names)):
        os.makedirs(os.path.join('../outputs', config['name'], img_names[c]), exist_ok=True)


    with torch.no_grad():
        for input, target, landmark_, meta in tqdm(val_loader, total=len(val_loader)):

            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)[0]
            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))
            print(iou)
            output = torch.sigmoid(output).cpu().numpy()
            target_ = target.cpu().numpy()
            
            output1 =  np.where(output > 0.5, 1, 0)
            FinalOutput = output1
            
            np.save('../outputs/{}/{}/PredIVDsAll.npy'.format(config['name'], meta['img_name'][0]), FinalOutput)
            
            FinalMask = target_
                
            FinalImgO = np.zeros((256, 128))
            FInalImgM = np.zeros((256, 128))
            FinalLandMark = np.zeros((256, 128))

            for k in range(12):
                FinalImgO += FinalOutput[0,k,:,:]
                FInalImgM += FinalMask[0,0,k,:,:]
            FinalImgO = (FinalImgO != 0)
            FInalImgM = (FInalImgM != 0)
            FinalImgO = (FinalImgO * 255).astype('uint8')
            FInalImgM = (FInalImgM * 255).astype('uint8')
            FinalLandMark = (FinalLandMark * 255).astype('uint8')
            PosAll = []
            # heatmap
            nums = 0
            for k in range(10,19):
                pos = landmark_[k][0][1:].cpu().numpy()
                x = pos[0]
                y = pos[1] + 64 - hashtable2[meta['img_name'][0]]
#                print(y)
                if not np.isnan(x) and not np.isnan(y):
                    nums += 1
                    PosAll.append([x,y,255])
                    
            hashtable[meta['img_name'][0]] = [nums, 0]
            cv2.imwrite(os.path.join('../outputs', config['name'], meta['img_name'][0],  'PredIVDs' + '.bmp'),
                            FinalImgO, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join('../outputs', config['name'], meta['img_name'][0], 'RealIVDs' + '.bmp'),
                            FInalImgM, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            map = heat_map(PosAll, 256, 128)
            cv2.imwrite(os.path.join('../outputs', config['name'], meta['img_name'][0], 'IVDsHeatmap' + '.bmp'),
                            map, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('IoU: %.4f' % avg_meter.avg)
    np.save('../outputs/{}/Loc.npy'.format(config['name']), hashtable)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
