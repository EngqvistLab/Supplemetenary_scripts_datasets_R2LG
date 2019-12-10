
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate
from hyperopt import hp
from Bio import SeqIO
import pandas as pd
import re

def to_binary(seq):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(20)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),20))
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(20))
        seq_coding[i,:] = code
    return seq_coding


def zero_padding(inp,length,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    assert len(inp) < length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp
    else:
        out[0:inp.shape[0]] = inp
    return out

def make_sets2(X1,X2,Y,split=0.9):
    '''create randomly shuffled indices for X and Y
    separate one hot vectors and values'''
    # improve storage efficiency
    np.random.seed(seed=10)
    ind1 = np.random.permutation(np.linspace(0,X1.shape[0]-1,X1.shape[0],dtype='int32'))
    splt = np.round(X1.shape[0]*split)-1
    splt = splt.astype('int64')
    X1_train = np.int8(X1[ind1[:splt]]) # one hot!!
    X1_test = np.int8(X1[ind1[splt:]])
    
    if X2 is not None:
        X2_train = X2[ind1[:splt]]
        X2_test = X2[ind1[splt:]]
    
    Y_train = Y[ind1[:splt]]
    Y_test = Y[ind1[splt:]]
    #names_train = names[ind1[:splt]]
    #names_test = names[ind1[splt:]]
    if X2 is None: return X1_train, X1_test, Y_train, Y_test
    else:return X1_train, X1_test, X2_train, X2_test, Y_train, Y_test #, names_train, names_test


def load_data(fname):
    ## load data, xseq for one-hot seqeunce features, y for temperatuer topt
    xseq,y = [],[]
    length_cutoff = 2000
    
    for rec in SeqIO.parse(fname,'fasta'):
        #>P43408 ogt=85;topt=70.0
        uni = rec.id
        ogt = float(rec.description.split()[-1].split(';')[0].split('=')[-1])
        topt = float(rec.description.split()[-1].split(';')[1].split('=')[-1])
        seq = rec.seq
        
        if len(seq)>length_cutoff: continue
        coding = to_binary(seq)
        coding = zero_padding(coding,length_cutoff)

        xseq.append(coding)
        
        ###### important to check to swith between topt and ogt 
        y.append(topt)
        ######

    xseq = np.array(xseq)
    y = np.array(y).reshape([len(y),1])

    print(xseq.shape,y.shape)
    X1_train, X1_test, Y_train, Y_test = make_sets2(xseq,None,y,split=0.9)
    print(X1_train.shape,X1_test.shape)
    print(Y_train.shape,Y_test.shape)

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))
    
    return X1_train, X1_test, Y_train, Y_test

def Params():
    params = {
        'kernel_size1': [20, 30, 40],
        'filters1': [32, 64],
        'dilation1': [1, 2, 4],
        'pool_size1': [2, 4, 8],
        'stride1': [2, 4, 8],
        'dropout1': (0, 0.4),
        
        'kernel_size2': [10, 20, 30],
        'filters2': [ 64, 128],
        'dilation2': [1, 2, 4],
        'pool_size2': [1, 2, 4],
        'stride2': [1, 2],
        'dropout2': (0, 0.4),
        
        'kernel_size3': [10, 20],
        'filters3': [128, 256],
        'dilation3': [1, 2, 4],
        'pool_size3': [1, 2, 4],
        'stride3': [1, 2],
        'dropout3': (0, 0.4),
        
        'dense5': [64, 128],
        'dropout5': (0, 0.3),
        
        'dense6': [32, 64],
        'dropout6': (0, 0.3)
    }
    return {k: hp.choice(k, v) if type(v) == list else hp.uniform(k, v[0], v[1]) for k, v in params.items()}


def POC_model(input_shape, p):
    input_shape_hot = input_shape[0]
    X_input1 = Input(shape = input_shape_hot)
    
    ## Step 1 - hot layers
    #### the first three cnn layers
    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout1']))(X)
    
    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout1']))(X)
    
    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout1']))(X)
    
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    
    
    #### the second three cnn layers
    X = Conv1D(filters=int(p['filters2']),kernel_size=int(p['kernel_size2']),strides=1,dilation_rate=int(p['dilation2']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)
    
    
    X = Conv1D(filters=int(p['filters2']),kernel_size=int(p['kernel_size2']),strides=1,dilation_rate=int(p['dilation2']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)
    
    
    X = Conv1D(filters=int(p['filters2']),kernel_size=int(p['kernel_size2']),strides=1,dilation_rate=int(p['dilation2']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size2']), strides=int(p['stride2']), padding='same')(X)
    
    
    
    #### the last three cnn layers
    X = Conv1D(filters=int(p['filters3']),kernel_size=int(p['kernel_size3']),strides=1,dilation_rate=int(p['dilation3']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout3']))(X)
    
    X = Conv1D(filters=int(p['filters3']),kernel_size=int(p['kernel_size3']),strides=1,dilation_rate=int(p['dilation3']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout3']))(X)
    
    X = Conv1D(filters=int(p['filters3']),kernel_size=int(p['kernel_size3']),strides=1,dilation_rate=int(p['dilation3']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout3']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size3']), strides=int(p['stride3']), padding='same')(X)
    
    
    # Step 2 - neural network merge data
    X = Flatten()(X)
    
    X = Dense(int(p['dense5']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout5']))(X)
    
    X = Dense(int(p['dense6']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout6']))(X)
    
    # Step 3 - output
    X = Dense(1)(X)
    model = Model(inputs = [X_input1], outputs = X)
    
    return model
