# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:39:45 2021

@author: Madmax
"""

import SimpleITK as sitk

import sys

import os

#file = r'H:\Competition\Spine_Segmentation\Spine_Segmentation_new_data\Case27\Mask_512.nii.gz'

file = r'H:\Competition\SpineSeg\code-self\dataset\TestData\Case28\raw_MR.nii.gz'
#file = r'H:\Competition\SpineSeg\code-self\dataset\RawData\Case156\raw_MR.nii.gz'
#file = r'H:\Competition\SpineSeg\code-self\dataset\RawData\Case153\Mask.nii.gz'
#file = r'H:\Competition\SpineSeg\code-self\segmentation_results_v2\segmentation_results\seg_case1.nii.gz'

#file = r'H:\Competition\SpineSeg\code-self\segmentation_results_v2\segmentation_results\seg'
#file = r'H:\Competition\SpineSeg\code-self\dataset\Test\Prediction\Case96\pred_heatmap.nii.gz'

#file = r'H:\Competition\SpineSeg\code-self\outputs\testdata\Case96\cropMR_256.nii.gz'

example = sitk.ReadImage(file)

sitk.Show(example)