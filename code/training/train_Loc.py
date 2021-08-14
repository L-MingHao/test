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
from network.SCnet import SpatialConfigurationNet
from network.archs import AttentionUNet, NestedUNet, ConvFour, R2UNet
from dataset.dataset import Dataset, Dataset2
from losses import LovaszLossSoftmax, LovaszLossHinge, dice_coeff

DATEINFO = datetime.datetime.now().strftime("%Y-%m-%d")

# 解析参数
def parse_args():
    parser = argparse.ArgumentParser()

    # train info 
    parser.add_argument('--name', default=None,
                        help='model name: (default: SpatialConfigurationNet)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    
    # model info 
    parser.add_argument('--model', '-a', metavar='ARCH', default='SpatialConfigurationNet')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--input_d', default=12, type=int,
                        help='image depth')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss info 
    parser.add_argument('--loss', default='MSE',
                        help='loss function')
    
    # dataset info
    parser.add_argument('--dataset', default='spinedata',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.nii.gz',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.nii.gz',
                        help='mask file extension')

    # optimizer info
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler info
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to train 
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output 
        output = model(input)
        
        loss = criterion(output, target)

        iou = (iou_score(output, target))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            iou = (iou_score(output, target))

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
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

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s_model' % (config["dataset"], config['model'])

    outdirs = DATEINFO + '-' + config["name"] + '-' + str(config["epochs"]) + '-' + config["loss"]
    os.makedirs("../models/%s" % outdirs, exist_ok = True)
    os.makedirs("../outputs/%s" % outdirs, exist_ok = True)
    writer = SummaryWriter("../outputs/%s" % outdirs)


    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('../models/%s/config.yml' % outdirs, 'w') as f:
        yaml.dump(config, f)

    # define Loss function
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'MSE':
        criterion = nn.MSELoss().cuda()
    elif config['loss'] == 'L1smooth':
        criterion = nn.SmoothL1Loss().cuda()
    elif config['loss'] == 'LovaszLoss':
        if config['num_classes'] > 1:
            criterion = LovaszLossSoftmax().cuda()
        else:
            criterion = LovaszLossHinge().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['model'])
    model = R2UNet(in_channels = 1)
    model = model.cuda()

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 指数衰减学习率
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.1, -1)
    else:
        raise NotImplementedError

    # Data loading code
    img_names = os.listdir('../outputs/data/')

    train_img_names, val_img_names = train_test_split(img_names, test_size=0.2, random_state=41)

    train_transform = None
    val_transform = None

    train_dataset = Dataset2(img_names=train_img_names,
                            img_dir='../outputs/data/',
                            transform=train_transform)
    val_dataset = Dataset2(img_names=val_img_names,
                        img_dir='../outputs/data/',
                        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False)

    # output log
    log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
    ])

    best_iou = 100
    trigger = 0

    # training 
    for epoch in range(config['epochs']):
        print("Epoch [ %d / %d ] " % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        # learning rate policy
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        writer.add_scalar("Train/Loss", train_log['loss'], epoch)
        writer.add_scalar("Train/Iou", train_log['iou'], epoch)
        writer.add_scalar("Val/Loss", val_log["loss"], epoch)
        writer.add_scalar("Val/Iou", val_log['iou'], epoch)

        # log save
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        pd.DataFrame(log).to_csv('../models/%s/log.csv' %
                                 outdirs, index=False)

        trigger += 1

        # save best model
        if val_log['loss'] < best_iou:
            torch.save(model.state_dict(), '../models/%s/model.pth' %
                       outdirs)
            best_iou = val_log['loss']
            print("=> saved best model")
            trigger = 0    

        # early stop
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break   

        torch.cuda.empty_cache()
    writer.export_scalars_to_json("./all_sclars.json")
    writer.close()

if __name__ == "__main__":
    main()