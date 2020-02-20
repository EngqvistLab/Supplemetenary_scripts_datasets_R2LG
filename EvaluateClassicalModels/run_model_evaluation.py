#!/usr/bin/env python
# coding: utf-8

# #### 1. try different regression models
# ##### Gang Li, 2018-09-21

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression as LR
from sklearn import svm
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
import sys
from multiprocessing import cpu_count

import os

def normalize(X):
    X_n = np.zeros_like(X)
    k = 0
    for i in range(X.shape[1]):
        x = X[:,i]
        if np.var(x) == 0: continue
        X_n[:,k] = (x-np.mean(x))/np.var(x)**0.5
        k+=1
    return np.around(X_n, decimals=3)




def do_cross_validation(X,y,model):
    scores = cross_val_score(model,X,y,scoring='r2',cv=5,n_jobs=20)
    res = str(np.mean(scores))+','+str(np.std(scores))+'\n'
    print(res)
    print(scores)
    return res




def lr():
    return LR()




def elastic_net():
    return ElasticNetCV(n_jobs=20)



def bayesridge():
    model = BayesianRidge()
    return model




def svr():
    parameters={
                'C':np.logspace(-5,10,num=16,base=2.0),
                'epsilon':[0,0.01,0.1,0.5,1.0,2.0,4.0]
                }
    svr = svm.SVR(kernel='rbf', gamma='auto')
    model = GridSearchCV(svr,parameters,n_jobs=cpu_count(),cv=3)
    return model



def tree():
    parameters={
                'min_samples_leaf':np.linspace(0.01,0.5,10)
                }
    dtr=DecisionTreeRegressor()
    model=GridSearchCV(dtr,parameters,n_jobs=cpu_count(),cv=3)
    return model



def random_forest():
    parameters = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    rf = RandomForestRegressor(n_estimators=1000,verbose =1,n_jobs=-1)
    model=GridSearchCV(rf,parameters,n_jobs=1,cv=3)
    return model


def test_model_performace(infile,outfile):
    df = pd.read_csv(infile,index_col=0)
    print(df.shape)
    df = df.round(3)
    X,y = df.values[:,:-1],df.values[:,-1]
    print(X.shape)
    X = normalize(X)
    print(X.shape)
   
    fhand = open(outfile,'w')
    fhand.write('model,mean,std\n')
    
    fhand.write('Linear model,')
    fhand.write(do_cross_validation(X,y,lr()))
    
    fhand.write('Elastic Net,')
    fhand.write(do_cross_validation(X,y,elastic_net()))
    
    fhand.write('BayesRige,')
    fhand.write(do_cross_validation(X,y,bayesridge()))
    
    fhand.write('SVR model,')
    fhand.write(do_cross_validation(X,y,svr()))
    
    fhand.write('Tree model,')
    fhand.write(do_cross_validation(X,y,tree()))

    fhand.write('Random forest,')
    fhand.write(do_cross_validation(X,y,random_forest()))
    
    fhand.close()


if __name__ == "__main__":
    infile = sys.argv[-2]
    outdir = sys.argv[-1]
    outfile = os.path.join(outdir,infile.split('/')[-1].replace('.csv','.out'))
    test_model_performace(infile,outfile)

