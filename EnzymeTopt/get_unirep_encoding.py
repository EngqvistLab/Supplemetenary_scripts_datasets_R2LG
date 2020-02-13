#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unirep
from Bio import SeqIO
import pandas as pd
import numpy as np
import sys

seqfile = sys.argv[-1]
outfile = seqfile+'.csv'

bab =  unirep.babbler1900(model_path='./1900_weights/')

data,index = [],[]
for rec in SeqIO.parse(seqfile,'fasta'):
    seq = rec.seq
    if not bab.is_valid_seq(seq): continue

    av, fh, fc = bab.get_rep(seq)
    lst = list(av)+list(fh)+list(fc)
    data.append(lst)
    index.append(rec.id)

data = np.array(data)
print(data.shape)

columns = ['av_{0}'.format(i+1) for i in range(1900)] + ['fh_{0}'.format(i+1) for i in range(1900)] + ['fc_{0}'.format(i+1) for i in range(1900)]
dfall = pd.DataFrame(data=data,index=index,columns=columns)

dfall.to_csv(outfile)
