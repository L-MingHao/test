import sys
sys.path.append("..")
import argparse
from glob import glob
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import autocast 
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from utils.utils import AverageMeter, str2bool
from online_evaluation import online_evaluation
from model import SpatialConfigurationNet
from loss import LovaszLossSoftmax, LovaszLossHinge, dice_coeff

import os
import sys
import random
import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

if os.path.abspath('../..') not in sys.path:
    sys.path.insert(0, os.path.abspath('../..'))

from utils.tools import csv_to_catalogue
from DataLoader.dataloader_IVDlocation import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer

DATEINFO = datetime.datetime.now().strftime("%Y-%m-%d")

if __name__ == '__main__':

    # added by ChenChen Hu
    print('This script has been modified by Madmax !')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=0,
                        help='list_GPU_ids for training (default: 0)')
    parser.add_argument('--max_iter',  type=int, default=50000,
                        help='training iterations(default: 50000)')
    # added by Chenchen Hu
    parser.add_argument('--num_classes', type=int, default=1)

    parser.add_argument('--catalogue', type=int, default=0)
    parser.add_argument('--latest', type=int, default=0,
                        help='load the latest model')
    parser.add_argument('--model_path', type=str, default='../../../Output/IVD_Location/latest.pkl')
    parser.add_argument('--model_type', type=str, default='SpatialConfigurationNet',
                        help='the type of model')
    parser.add_argument('--seed', type=int, default=68,
                        help='set the seed')

    parser.add_argument('--loss', default='LovaszLoss',
                        help='loss function')

    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    torch.manual_seed(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'IVD_Location'
    trainer.setting.output_dir = '../../../Output/IVD_Location'
    list_GPU_ids = args.list_GPU_ids
    csv_path = '../../Catalogue' + '/' + str(args.catalogue) + '.csv'
    catalogue = csv_to_catalogue(csv_path)

    # setting.network is an object
    if args.model_type == 'SpatialConfigurationNet':
        trainer.setting.network = SpatialConfigurationNet(num_labels=20)
        print('Loading SpatialConfigurationNet!')

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader = get_loader(
        catalogue=catalogue,
        batch_size=args.batch_size,  # 2
        num_samples_per_epoch=args.batch_size * 500,
        num_works=4,
        phase='train'
    )

    trainer.setting.val_loader = get_loader(
        catalogue=catalogue,
        batch_size=1,
        num_samples_per_epoch=len(list(catalogue['val'].dropna())),
        num_works=4,
        phase='val'
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    if args.loss == 'BCEWithLogitsLoss':
        trainer.setting.loss_function = nn.BCEWithLogitsLoss()
    elif args.loss == 'MSE':
        trainer.setting.loss_function = nn.MSELoss()
    elif args.loss == 'L1smooth':
        trainer.setting.loss_function = nn.SmoothL1Loss()
    elif args.loss == 'LovaszLoss':
        if args.num_classes > 1:
            trainer.setting.loss_function = LovaszLossSoftmax()
        else:
            trainer.setting.loss_function = LovaszLossHinge()
    trainer.setting.online_evaluation_function_val = online_evaluation

    # optimizer
    params = filter(lambda p: p.requires_grad, trainer.setting.network.parameters())
    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 1e-3,
                              'weight_decay': 1e-4,
                              'momentum': 0.9,
                              'nesterov': False
                          },
                          params = params
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             cfgs={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)

    # added by Chenchen Hu
    # load the latest model when the recovery is True and the model exists.
    if args.latest and os.path.exists(args.model_path):
        trainer.init_trainer(ckpt_file=args.model_path,
                             list_GPU_ids=list_GPU_ids,
                             only_network=False)

    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')