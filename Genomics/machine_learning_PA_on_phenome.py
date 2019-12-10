import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def do_cross_validation(X,y,model):
    scores = cross_val_score(model,X,y,scoring='r2',cv=5,n_jobs=-1)
    return [np.mean(scores),np.std(scores)]

def random_forest():
    parameters = {
                    'min_samples_leaf':np.linspace(0.01,0.5,10)
    }
    rf = RandomForestRegressor(n_estimators=1000,n_jobs=-1,max_features='sqrt')
    model=GridSearchCV(rf,parameters,n_jobs=-1) 
    return model

infile = 'data/gene_presence_absence_v4.tsv'
df = pd.read_csv(infile,sep='\t',index_col=0)

dfph = pd.read_csv('data/phenoMatrix_35ConditionsNormalizedByYPD_indexed.tab',sep='\t',index_col=0)

results = list()
for col in dfph.columns:
    X = df.loc[dfph.index,:].values
    X[X=='Y'] = 1
    X[X=='N'] = 0
    y = dfph[col]
    print(X.shape,y.shape)
    
    # remove nans in y
    X = X[~np.isnan(y),:]
    y = y[~np.isnan(y)]
    
    res = do_cross_validation(X,y,random_forest())
    print(col,res)
    results.append(res)

dfres = pd.DataFrame(data=results,index=dfph.columns,columns=['r2','r2_std'])
dfres.to_csv('results/cross_validation_results_PA_35_phenotypes.csv')