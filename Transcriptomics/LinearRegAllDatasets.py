#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:46:03 2019

@author: azams

All possible variants of the datasets created by *PrepareDataSetVariants.py* 
are loaded one by one and cross validated results using **LinearRegression** 
are produced and saved as an excel sheet.
This forms the baseline results for the more sophisticated regression models.
Since, in some cases the nubmer of features grow closer to 4k, an option to 
apply PCA based dimentionality reduction is also featured. 
For this, set pcaScaling = True.
"""

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# LoadDataCSV is a py file, to be present in the current working directory.
# LoadDataSet is a function defined in it which is used here for laoding 
# required data files step by step.
from LoadDataCSV import LoadDataSet

# PCA scaling is set to False by default
pcaScaling = False
#%% LinearRegression applied to the entire variants of the data

rsd = ['rsd1', 'rsd2', 'rsd3'] #r
target = ['Yc', 'Yu'] #t
Ks = ['K4', 'K5', 'K5s', 'K6'] #k
features = ['codon', 'stability', 'kmers', 'codon_stability', 'codon_kmers', 'codon_stability_kmers', 'stability_kmers'] 

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Log transformation to apply on the features
transformer = FunctionTransformer(np.log1p, validate=True)
pca = PCA(n_components=0.95, svd_solver = 'full')
lr = LinearRegression()

if pcaScaling:
    model = make_pipeline(transformer, pca, lr)
else:
    model = make_pipeline(transformer, lr)

res = []
scoring = ['neg_mean_squared_error', 'r2']

for r in rsd:
    for t in target:
        for k in Ks:
            for f in features:
                X, y = LoadDataSet(r,t,k,f)
                print('\n'+ r + ' ' + t+ ' ' +k+ ' ' +f)
                scores = cross_validate(model, X, y, cv=cv, scoring= scoring, n_jobs=-1, return_train_score=True)
                sorted(scores.keys())
                test_mse = -1. * (scores['test_neg_mean_squared_error'].mean())
                test_r2 = scores['test_r2'].mean()
                train_mse = -1. * (scores['train_neg_mean_squared_error'].mean())
                train_r2 = scores['train_r2'].mean()
                fit_time = scores['fit_time'].mean()
                score_time = scores['score_time'].mean()
                print("train_r2: {:.2}\n test_r2: {:.2}\n train_mse: {:.2}\n test_mse: {:.2}\n fit_time: {:.2}\n score_time: {:.2}\n".format(train_r2, test_r2, train_mse, test_mse, fit_time, score_time) )
                res.append([r,t,k,f,train_r2, test_r2, train_mse, test_mse, fit_time, score_time] )
    np.save(r+t+k,res)
#%% Write the results to the Excel file
dfres = pd.DataFrame(res)
dfres.columns = columns=['rsd', 'target', 'K', 'features', 'train_r2', 'test_r2', 'train_mse', 'test_mse', 'fit_time', 'score_time']
if pcaScaling:
    writer = pd.ExcelWriter(os.getcwd() + '/Results/LinearRegression_CV_results_AllData_PCA.xlsx')
else:
    writer = pd.ExcelWriter(os.getcwd() + '/Results/LinearRegression_CV_results_AllData.xlsx')
    
# write as a sheet in the excel 
dfres.to_excel(writer, sheet_name= 'LogTransform_LR', index=False)
writer.save()
print("Cross Validated Linear REgression results on the entire datasets have been saved in the xlsx file.")
