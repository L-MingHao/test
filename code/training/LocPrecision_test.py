# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:55:17 2021

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

import sklearn
from sklearn.model_selection import train_test_split
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.metrics as sm
from sklearn import svm, ensemble, tree, neighbors

if __name__ == '__main__':
    modelAll = []
    img_names = os.listdir('../outputs/data/')[:-3]
    w = 32
    h = 48
    
    img_names.remove("Case39")
    img_names.remove("Case12")
#    img_names.remove("Case91")
    d = dict()
    for i in range(len(img_names)):
        d[img_names[i]] = [None for _ in range(19)]
    
    test_names = os.listdir("../outputs/testdata/")[:-1]
    
    for target in ['x', 'y']:
        print("target:{}".format(target))
        print("********************************")
        x = [[] for _ in range(10)]
        y = [[] for _ in range(10)]
        xr = [[] for _ in range(10)]
        
        hashtable = dict()
        hashtabletest = dict()
        if target == 'x':
            hashtableVer = dict()
        for index in range(len(img_names)):
            path = "../outputs/data/" + img_names[index] + '/PredPoints.csv'
            path2 = "../outputs/data/" + img_names[index] + '/RealPoints.csv'
            path3 = "../outputs/data/" + img_names[index] + '/PredPointsVer.csv'
            pred = pd.read_csv(path3)
            predIvds = pd.read_csv(path)
            real = pd.read_csv(path2)
            
            hashtable[img_names[index]] = [[] for _ in range(10)]

            for i in range(len(pred)):

                if i == len(pred) - 1:
                    if i < len(predIvds) and not np.isnan(real[target][i]) and not np.isnan(real[target][i]) and not np.isnan(predIvds[target][i-1]) and not np.isnan(predIvds[target][i]):
                        x[i].append([pred[target][i], predIvds[target][i-1], predIvds[target][i]])
                        hashtable[img_names[index]][i] = [pred[target][i], predIvds[target][i-1], predIvds[target][i]]
                        y[i].append(real[target][i])
                    elif i == 9 and not np.isnan(real[target][i]) and not np.isnan(real[target][i]) and not np.isnan(predIvds[target][i-2]) and not np.isnan(predIvds[target][i-1]):
                        x[i].append([pred[target][i], predIvds[target][i-2], predIvds[target][i-1]])
                        hashtable[img_names[index]][i] = [pred[target][i], predIvds[target][i-2], predIvds[target][i-1]]
                        y[i].append(real[target][i])                        
                elif i == 0:
                    if  not np.isnan(real[target][i]) and not np.isnan(real[target][i]) and not np.isnan(predIvds[target][i]) and not np.isnan(predIvds[target][i+1]):
                        x[i].append([pred[target][i], predIvds[target][i], predIvds[target][i+1]])
                        hashtable[img_names[index]][i] = [pred[target][i], predIvds[target][i], predIvds[target][i+1]]
                        y[i].append(real[target][i])                    
                elif  not np.isnan(real[target][i]) and not np.isnan(real[target][i]) and not np.isnan(predIvds[target][i-1]) and not np.isnan(predIvds[target][i]):
                    x[i].append([pred[target][i], predIvds[target][i-1], predIvds[target][i]])
                    hashtable[img_names[index]][i] = [pred[target][i], predIvds[target][i-1], predIvds[target][i]]
                    y[i].append(real[target][i])


        for index in range(len(test_names)):
            path = "../outputs/testdata/" + test_names[index] + '/PredPoints.csv'
            path3 = "../outputs/testdata/" + test_names[index] + '/PredPointsVer.csv'
            pred = pd.read_csv(path3)
            predIvds = pd.read_csv(path)
            
            hashtabletest[test_names[index]] = [[] for _ in range(10)]
            if target == 'x':
                hashtableVer[test_names[index]] = [[] for _ in range(19)]

            for i in range(len(pred)):

                if i == len(pred) - 1:
                    if i < len(predIvds) and not np.isnan(predIvds[target][i-1]) and not np.isnan(predIvds[target][i]):
                        hashtabletest[test_names[index]][i] = [pred[target][i], predIvds[target][i-1], predIvds[target][i]]
                    elif i == 9 and not np.isnan(real[target][i]) and not np.isnan(real[target][i]) and not np.isnan(predIvds[target][i-2]) and not np.isnan(predIvds[target][i-1]):
                        hashtabletest[test_names[index]][i] = [pred[target][i], predIvds[target][i-2], predIvds[target][i-1]]                    
                elif i == 0:
                    if  not np.isnan(predIvds[target][i]) and not np.isnan(predIvds[target][i+1]):
                        hashtabletest[test_names[index]][i] = [pred[target][i], predIvds[target][i], predIvds[target][i+1]]               
                elif  not np.isnan(predIvds[target][i-1]) and not np.isnan(predIvds[target][i]):
                    hashtabletest[test_names[index]][i] = [pred[target][i], predIvds[target][i-1], predIvds[target][i]]
            
            for i in range(len(pred)):
                if target == 'x':
                    hashtableVer[test_names[index]][i] = [pred[target][i], 0]
                else:
                    hashtableVer[test_names[index]][i][1] = pred[target][i]
            for i in range(len(predIvds)):
                if target == 'x':
                    hashtableVer[test_names[index]][i+10] = [predIvds[target][i], 0]
                else:
                    hashtableVer[test_names[index]][i+10][1] = predIvds[target][i]
                     
        model = pl.make_pipeline(
            sp.PolynomialFeatures(1),  # 多项式特征拓展器
            lm.LinearRegression()  # 线性回归器
        )
        
        train = [[] for _ in range(10)]
        for i in range(10):
            for j in range(len(x[i])):
                train[i].append([x[i][j],y[i][j]])
                
        
        d = [[] for _ in range(19)]
        
        for i in range(10):
            train_x = train[i]
            
            x = []
            y = []
#            tx = []
#            ty = []
            
            
            for j in range(len(train_x)):
                x.append(train_x[j][0])
                y.append(train_x[j][1])
            
            
            x = np.array(x)
            y = np.array(y)
    
    
            x = x.reshape(-1, 3)
            model.fit(x, y)
            modelAll.append(model)
            
#            tx = tx.reshape(-1, 3)
    
#            pred_y = model.predict(tx)

            for j in range(len(test_names)):
                xrr = np.array(hashtabletest[test_names[j]][i])
                xrr = xrr.reshape(-1, 3)
                if len(xrr) != 0:
                    if target == 'x':
                        hashtableVer[test_names[j]][i] = [float(model.predict(xrr)),0]
                    else:
                        hashtableVer[test_names[j]][i][1] = float(model.predict(xrr))
    
    hashtableHeat = np.load('../outputs/testdata/LocPos.npy', allow_pickle=True).item()
    for i in hashtableVer:
        hashtableIVDsZ = np.load("../outputs/testdata/{}/PredIVDsAll.npy".format(i))
        hashtableVerZ = np.load("../outputs/testdata/{}/PredVerAll.npy".format(i))
        
        for j in range(10):
            if hashtableVer[i][j] != []:
                temp = 0
                for k in range(len(hashtableVerZ)):
                    temp += (k) * np.sum(hashtableVerZ[k,max(0, int(hashtableVer[i][j][0]-36)):min(256,int(hashtableVer[i][j][0]+36)),max(0, int(hashtableVer[i][j][1]-36- hashtableHeat[i] + 32)):min(128,int(hashtableVer[i][j][1])+36- hashtableHeat[i] + 32)])
                print(np.sum(hashtableVerZ[:,max(0, int(hashtableVer[i][j][0]-36)):min(256,int(hashtableVer[i][j][0]+36)),max(0, int(hashtableVer[i][j][1]-36+ hashtableHeat[i] - 128)):min(128,int(hashtableVer[i][j][1])+36+ hashtableHeat[i] - 128)]))
                hashtableVer[i][j].append(temp / np.sum(hashtableVerZ[:,max(0, int(hashtableVer[i][j][0]-36)):min(256,int(hashtableVer[i][j][0]+36)),max(0, int(hashtableVer[i][j][1]-36- hashtableHeat[i] + 32)):min(128,int(hashtableVer[i][j][1])+36- hashtableHeat[i] + 32)]))
                
        for j in range(10,19):
            if hashtableVer[i][j] != []:
                temp = 0
                for k in range(len(hashtableIVDsZ)):
                    temp += (k) * np.sum(hashtableIVDsZ[k,max(0, int(hashtableVer[i][j][0])-36):min(256,int(hashtableVer[i][j][0])+36),max(0, int(hashtableVer[i][j][1])-36- hashtableHeat[i] + 32):min(128,int(hashtableVer[i][j][1])+36- hashtableHeat[i] + 32)])
                hashtableVer[i][j].append(temp / np.sum(hashtableIVDsZ[:,max(0, int(hashtableVer[i][j][0])-36):min(256,int(hashtableVer[i][j][0])+36),max(0, int(hashtableVer[i][j][1])-36 - hashtableHeat[i] + 32):min(128,int(hashtableVer[i][j][1])+36 - hashtableHeat[i] + 32)]))
            
    for i in hashtableVer:
        for j in range(len(hashtableVer[i])):
            if hashtableVer[i][j] != []:
                hashtableVer[i][j][0] *= 2
                hashtableVer[i][j][1] *= 2
        save = pd.DataFrame(hashtableVer[i])
        save.columns = ['y', 'x', 'z']
        save[['z','y','x']].T.to_csv('../outputs/testdata/{}/Predfinal.csv'.format(i),index=True,header=True)
        