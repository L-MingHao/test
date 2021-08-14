# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:45:40 2021

@author: Madmax
"""
import matplotlib.pyplot as plt
import pandas as pd 
import SimpleITK as sitk
import cv2
import numpy as np
img_names = 'Case28'
path = "../outputs/testdata/" + img_names + '/Predfinal.csv'
pathi = "../dataset/TestData/" + img_names + '/MR_512.nii.gz'
pred = pd.read_csv(path)
img = sitk.ReadImage(pathi, sitk.sitkFloat32)
img = sitk.GetArrayFromImage(img)

w = 24
h = 48

for j in range(len(img)):
    for i in pred:
        if i == 'Unnamed: 0':
            pass
        else:
            if not np.isnan(pred[i][1]) and int(i) < 10:
                cv2.rectangle(img[j,:,:], (max(0, int(pred[i][2]-h)), max(0, int(pred[i][1]-w))), (min(512,int(pred[i][2]+h)), min(512, int(pred[i][1]+w))), 255, 2) 
    #plt.matshow(img[5,:,:], cmap=plt.cm.gray)

#sitk.WriteImage(img,'simpleitk_save.nii.gz')

sitk.Show(sitk.GetImageFromArray(img))