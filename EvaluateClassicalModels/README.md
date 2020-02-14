### Description

#### Usage:  
```
python run_model_evaluation.py infile.csv outdir
```

`infile.csv` is an input comma-sperated file with first column as index and last column is the target column. Other columns are the features. The script will firstly standardize each column and then test the performance of six different regression models via a nested cross validation approach. Those six models are:
```python
sklearn.linear_model.LinearRegression
sklearn.linear_model.BayesianRidge
sklearn.linear_model.ElasticNetCV
sklearn.tree.DecisionTreeRegressor
sklearn.svm.SVR
sklearn.ensemble.RandomForestRegressor
```

`outdir` is the directory where output file will be created. The name of the outputfile is `infile.out`. It is a csv file contains the mean and standard deviation of R2 score in the cross validation, for example:
```
model,mean,std
Linear model,0.5606470243036312,0.05182762180825625
Elastic Net,0.5561527172818607,0.046980868519725986
BayesRige,0.5613494374587507,0.051039246940720234
SVR model,0.547889662217659,0.026893484452659596
Tree model,0.5355491267446859,0.03453184980344376
Random forest,0.6021454315518417,0.028483260693779797
```

#### Computation time
Depending on the number of features, the script would take minutes to several hours.
