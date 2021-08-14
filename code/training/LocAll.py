# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:55:17 2021

@author: Madmax
"""

#import sys
#sys.path.append("..")
#import argparse
#from glob import glob
#import os
#from collections import OrderedDict
#
#import pandas as pd
#import numpy as np
#
#import sklearn
#from sklearn.model_selection import train_test_split
#import sklearn.pipeline as pl
#import sklearn.linear_model as lm
#import sklearn.preprocessing as sp
#import sklearn.metrics as sm
#from sklearn import svm, ensemble, tree, neighbors
#
#if __name__ == '__main__':
#    modelAll = []
#    img_names = os.listdir('../outputs/data/')[:-2]
#    w = 32
#    h = 48
#    
##    img_names.remove("Case208")
#    img_names.remove("Case91")
#    d = dict()
#    reald = dict()
#    for i in range(len(img_names)):
#        d[img_names[i]] = [None for _ in range(19)]
#    for target in ['x', 'y']:
#        print("target:{}".format(target))
#        print("********************************")
#        x = [[] for _ in range(10)]
#        y = [[] for _ in range(10)]
#        xr = [[] for _ in range(10)]
#        hashtable = dict()
#        if target == 'x':
#            hashtableVer = dict()
#        for index in range(len(img_names)):
#            path = "../outputs/data/" + img_names[index] + '/Fianl.csv'
#            path2 = "../outputs/data/" + img_names[index] + '/maskMid.csv'
#            pred = pd.read_csv(path)
#            real = pd.read_csv(path2)
#            hashtable[img_names[index]] = [[] for _ in range(10)]
#            if target == 'x':
#                hashtableVer[img_names[index]] = [[] for _ in range(19)]
#                reald[img_names[index]] = [[real['x'][l], real['y'][l]] for l in range(19)]
#            
#            for i in range(len(pred)):
#                if np.isnan(real['x'][i]) or np.isnan(real['y'][i]) or (i == 8 and (np.isnan(real['y'][i+1]) or np.isnan(real['x'][i+1]))):
#                    continue
#                else:
#                    if i == 0:
#                        temp = []
#                        for j in range(9):
#                            if j >= len(pred[target]):
#                                temp.append(0)
#                            else:
#                                temp.append(pred[target][j])
#                        x[i].append(temp)
#                        y[i].append(real[target][i])
#                    elif i == 8:
#                        temp = []
#                        for j in range(9):
#                            if j >= len(pred[target]):
#                                temp.append(0)
#                            else:
#                                temp.append(pred[target][j])
#                        x[i].append(temp)
#                        y[i].append(real[target][i])
#                        x[i+1].append(temp)
#                        y[i+1].append(real[target][i+1])
#                    else:
#                        temp = []
#                        for j in range(9):
#                            if j >= len(pred[target]):
#                                temp.append(0)
#                            else:
#                                temp.append(pred[target][j])
#                        x[i].append(temp)
#                        y[i].append(real[target][i])
#            
#            for i in range(len(pred)):
#                if target == 'x':
#                    hashtableVer[img_names[index]][i+10] = [pred[target][i], 0]
#                else:
#                    hashtableVer[img_names[index]][i+10][1] = pred[target][i]
#                if i == 0:
#                    temp = []
#                    for j in range(9):
#                        if j >= len(pred[target]):
#                            temp.append(0)
#                        else:
#                            temp.append(pred[target][j])
#                    hashtable[img_names[index]][i] = temp
#                elif i == 8:
#                    temp = []
#                    for j in range(9):
#                        if j >= len(pred[target]):
#                            temp.append(0)
#                        else:
#                            temp.append(pred[target][j])
#                    hashtable[img_names[index]][i] = temp
#                    hashtable[img_names[index]][i+1] = temp
#                else:
#                    temp = []
#                    for j in range(9):
#                        if j >= len(pred[target]):
#                            temp.append(0)
#                        else:
#                            temp.append(pred[target][j])
#                    hashtable[img_names[index]][i] = temp
##
##            for i in range(len(pred)):
##                if np.isnan(real[target][i+1]) or (i == 0 and (np.isnan(real[target][i+1]) or np.isnan(real[target][i]))):
##                    continue
##                else:
##                    if i == 0:
##                        x[i].append(pred[target][i])
##                        y[i].append(real[target][i])
##                        x[i+1].append(pred[target][i])
##                        y[i+1].append(real[target][i+1])
##                    else:
##                        x[i+1].append(pred[target][i])
##                        y[i+1].append(real[target][i+1])
##            
##            for i in range(len(pred)):
##                if target == 'x':
##                    hashtableVer[img_names[index]][i+10] = [pred[target][i], 0]
##                else:
##                    hashtableVer[img_names[index]][i+10][1] = pred[target][i]
##                if i == 0:
##                    hashtable[img_names[index]][i] = pred[target][i]
##                    hashtable[img_names[index]][i+1] = pred[target][i]
##                else:
##                    hashtable[img_names[index]][i+1] = pred[target][i]
#
#
#        model = pl.make_pipeline(
#            sp.PolynomialFeatures(1),  # 多项式特征拓展器
#            lm.LinearRegression()  # 线性回归器
#        )
#        print("haha")
#        
#    #    model = ensemble.AdaBoostRegressor(n_estimators=10)
#    #    model = tree.DecisionTreeRegressor()
#        
##        model = neighbors.KNeighborsRegressor(weights="uniform")
##        model = svm.SVR(kernel="rbf")
#        
#        train = [[] for _ in range(10)]
#        for i in range(10):
#            for j in range(len(x[i])):
#                train[i].append([x[i][j],y[i][j]])
#        
#        d = [[] for _ in range(19)]
#        
#        for i in range(10):
#            train_x, val_x = train_test_split(train[i], test_size=0.2, random_state=41)
#            
#            x = []
#            y = []
#            tx = []
#            ty = []
#            
#            
#            for j in range(len(train_x)):
#                x.append(train_x[j][0])
#                y.append(train_x[j][1])
#            
#            for j in range(len(val_x)):
#                tx.append(val_x[j][0])
#                ty.append(val_x[j][1])            
#            
#            
#            x = np.array(x)
#            y = np.array(y)
#            
#            tx = np.array(tx)
#            ty = np.array(ty)        
#    
#    
#            x = x.reshape(-1, 9)
#            model.fit(x, y)
#            modelAll.append(model)
#            
#            tx = tx.reshape(-1, 9)
#    
#        #    
#            pred_y = model.predict(tx)
#            
#            for j in range(len(img_names)):
#                xrr = np.array(hashtable[img_names[j]][i])
#                xrr = xrr.reshape(-1, 9)
#                if len(xrr) != 0:
#                    if target == 'x':
#                        hashtableVer[img_names[j]][i] = [float(model.predict(xrr)),0]
#                    else:
#                        hashtableVer[img_names[j]][i][1] = float(model.predict(xrr))
#
#        #   
#            print(pred_y)
#            print(ty)
#            # 模型评估
#            print("{}".format(i))
#            print('平均绝对值误差：', sm.mean_absolute_error(ty, pred_y))
#            print('平均平方误差：', sm.mean_squared_error(ty, pred_y))
#            print('中位绝对值误差：', sm.median_absolute_error(ty, pred_y))
#            print('R2得分：', sm.r2_score(ty, pred_y))
#            print('最大误差：', sm.max_error(ty, pred_y))
#            
#        
#    save = pd.DataFrame(hashtableVer)
#    save.to_csv("pred.csv",index=True,header=True)
#    save = pd.DataFrame(reald)
#    save.to_csv("real.csv",index=True,header=True)
#    
    
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
    img_names = os.listdir('../outputs/data/')[:-2]
    w = 32
    h = 48
    
#    img_names.remove("Case208")
    img_names.remove("Case91")
    d = dict()
    reald = dict()
    for i in range(len(img_names)):
        d[img_names[i]] = [None for _ in range(19)]
    for target in ['x', 'y']:
        print("target:{}".format(target))
        print("********************************")
        x = [[] for _ in range(10)]
        y = [[] for _ in range(10)]
        xr = [[] for _ in range(10)]
        hashtable = dict()
        if target == 'x':
            hashtableVer = dict()
        for index in range(len(img_names)):
            path = "../outputs/data/" + img_names[index] + '/Fianl.csv'
            path2 = "../outputs/data/" + img_names[index] + '/maskMid.csv'
            pred = pd.read_csv(path)
            real = pd.read_csv(path2)
            hashtable[img_names[index]] = [[] for _ in range(10)]
            if target == 'x':
                hashtableVer[img_names[index]] = [[] for _ in range(19)]
                reald[img_names[index]] = [[real['x'][l], real['y'][l]] for l in range(19)]
            

            for i in range(len(pred)):
                if np.isnan(real[target][i+1]) or (i == 0 and (np.isnan(real[target][i+1]) or np.isnan(real[target][i]))):
                    continue
                else:
                    if i == 0:
                        x[i].append(pred[target][i])
                        y[i].append(real[target][i])
                        x[i+1].append(pred[target][i])
                        y[i+1].append(real[target][i+1])
                    else:
                        x[i+1].append(pred[target][i])
                        y[i+1].append(real[target][i+1])
            
            for i in range(len(pred)):
                if target == 'x':
                    hashtableVer[img_names[index]][i+10] = [pred[target][i], 0]
                else:
                    hashtableVer[img_names[index]][i+10][1] = pred[target][i]
                if i == 0:
                    hashtable[img_names[index]][i] = pred[target][i]
                    hashtable[img_names[index]][i+1] = pred[target][i]
                else:
                    hashtable[img_names[index]][i+1] = pred[target][i]


        model = pl.make_pipeline(
            sp.PolynomialFeatures(1),  # 多项式特征拓展器
            lm.LinearRegression()  # 线性回归器
        )
        print("haha")
        
    #    model = ensemble.AdaBoostRegressor(n_estimators=10)
    #    model = tree.DecisionTreeRegressor()
        
#        model = neighbors.KNeighborsRegressor(weights="uniform")
#        model = svm.SVR(kernel="rbf")
        
        train = [[] for _ in range(10)]
        for i in range(10):
            for j in range(len(x[i])):
                train[i].append([x[i][j],y[i][j]])
        
        d = [[] for _ in range(19)]
        
        for i in range(10):
            train_x, val_x = train_test_split(train[i], test_size=0.2, random_state=41)
            
            x = []
            y = []
            tx = []
            ty = []
            
            
            for j in range(len(train_x)):
                x.append(train_x[j][0])
                y.append(train_x[j][1])
            
            for j in range(len(val_x)):
                tx.append(val_x[j][0])
                ty.append(val_x[j][1])            
            
            
            x = np.array(x)
            y = np.array(y)
            
            tx = np.array(tx)
            ty = np.array(ty)        
    
    
            x = x.reshape(-1, 1)
            model.fit(x, y)
            modelAll.append(model)
            
            tx = tx.reshape(-1, 1)
    
        #    
            pred_y = model.predict(tx)
            
            for j in range(len(img_names)):
                xrr = np.array(hashtable[img_names[j]][i])
                xrr = xrr.reshape(-1, 1)
                if len(xrr) != 0:
                    if target == 'x':
                        hashtableVer[img_names[j]][i] = [float(model.predict(xrr)),0]
                    else:
                        hashtableVer[img_names[j]][i][1] = float(model.predict(xrr))

        #   
            print(pred_y)
            print(ty)
            # 模型评估
            print("{}".format(i))
            print('平均绝对值误差：', sm.mean_absolute_error(ty, pred_y))
            print('平均平方误差：', sm.mean_squared_error(ty, pred_y))
            print('中位绝对值误差：', sm.median_absolute_error(ty, pred_y))
            print('R2得分：', sm.r2_score(ty, pred_y))
            print('最大误差：', sm.max_error(ty, pred_y))
            
    
    savep = pd.read_csv('pred.csv')
    
    for i in savep.keys():
        if i in hashtableVer:
            for j in range(19):
                if savep[i][j] == '[]':
                    savep[i][j] = hashtableVer[i][j]
    
#    save = pd.DataFrame(hashtableVer)
    savep.to_csv("pred.csv",index=True,header=True)