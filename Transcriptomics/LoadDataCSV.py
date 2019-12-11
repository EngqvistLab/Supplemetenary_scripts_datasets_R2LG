#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:46:03 2019

@author: azams

This file provides a simple code which makes it easier to load a required 
data.csv data file out of the data directories created using 
PrepareDataSetVariants.py,

Two possibilities:
i) plain code, read the required data.csv, randomly permutes the rows of the
   data and separates the X and y for any further analysis on it.

ii) The code is also embodied as a function 'LoadDataCSV' to load the required 
    data set  and returns X,y. Quicker to use inside other code files.
"""

import pandas as pd
import os
#%% Choose the right directory structure

# Possible names which are combined together to identify a unique data variant
# are defined as lists below
rsd = ['rsd1', 'rsd2', 'rsd3'] #r
target = ['Yc', 'Yu'] #y
Ks = ['K4', 'K5', 'K5s', 'K6'] #k
features = ['codon', 'stability', 'kmers', 'codon_stability', 'codon_kmers', 'codon_stability_kmers', 'stability_kmers'] #f

# Choose appropriate indices for 
        #   r, y, k, f
selector = [2, 0, 0, 0]
# to form a selector which we use below to come up with corresponding 
# directory names where the required data.csv resides.
parentDirName = rsd[selector[0]]+target[selector[1]]+Ks[selector[2]]

subDirName = features[selector[3]]

#%% Now load the data

dirPrefix = os.getcwd() + "/DataDirsByPrepareDataVariants/"
#dirPrefix = pathCWD + "/DataDirsByPrepareDataVariants/"
path = dirPrefix + parentDirName + "/" + subDirName + "/" 

# Read in data and put features matrix in X and target variable in y; both 
# numpy arrays of integers or floats
# last column of data.csv contains the target values

df = pd.read_csv(path+'data.csv')

# Randomly permure the rows and reindex again to avoid any possible order
df = df.sample(frac=1, axis=0, random_state=0).reset_index(drop=True)
array = df.values
X = array[:,0:len(df.columns)-1]
y = array[:,-1]

#%% Here we give another variant of loading data where we explictily pass the  
# requied values of:
# rsd: 'rsd1', 'rsd2', 'rsd3'
# target: 'Yc', 'Yu'
# K: 'K4', 'K5', 'K5s', 'K6'
# features: 'codon', 'stability', 'kmers', 'codon_stability', 'codon_kmers', 'codon_stability_kmers', 'stability_kmers'

def LoadDataSet(rsd= 'rsd2', target= 'Yc', K= 'K4', features= 'codon'):
    '''
    Possible values of:
     -rsd: 'rsd1', 'rsd2', 'rsd3'
     -target: 'Yc', 'Yu'
     -K: 'K4', 'K5', 'K5s', 'K6'
     -features: 'codon', 'stability', 'kmers', 'codon_stability', 'codon_kmers', 'codon_stability_kmers', 'stability_kmers'
    '''
    parentDirName = rsd + target + K    
    subDirName = features
    
    # Now load the data
    
    dirPrefix = os.getcwd() + "/DataDirsByPrepareDataVariants/"
    path = dirPrefix + parentDirName + "/" + subDirName + "/" 
    
    # Read in data and put features matrix in X and target variable in y; both 
    # numpy arrays of integers or floats
    # last column of data.csv contains the target values
    
    df = pd.read_csv(path+'data.csv')
    
    # Randomly permure the rows and reindex again to avoid any possible order
    df = df.sample(frac=1, axis=0, random_state=0).reset_index(drop=True)
    array = df.values
    X = array[:,0:len(df.columns)-1]
    y = array[:,-1]
    
    return X, y