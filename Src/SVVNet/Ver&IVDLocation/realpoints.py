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

import SimpleITK as sitk


img_names = os.listdir('../../../Data/Spine_Segmentation')[:-3]
mask_dir = '../../../Data/Spine_Segmentation'

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


    save = pd.DataFrame(d, columns = ['x', 'y'])
    if not os.path.exists('../../../Output/RealPoints/'+ img_names[i]):
        os.mkdir('../../../Output/RealPoints/'+ img_names[i])
    save.to_csv("../../../Output/RealPoints/{}/RealPoints.csv".format(img_names[i]),index=True,header=True)