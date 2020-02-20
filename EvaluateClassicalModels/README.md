### Description
This script evaluates six classical machine learning algorithms on feature-sets obtained for protein sequences in the Topt dataset.


#### Datasets
The scripts rely on an input file `infile.csv`, which should be comma-separated with first column as index and last column is the target column, i.e. the response variable (Topt). Other columns are the features. Depending on the analysis being carried out these will be from iFeature or UniRep embeddings. The file used for analysis in the paper is *not* provided here due to space constraints, but can easily be generated with scripts and data in the `EnzymeTopt` folder. However, to allow users to quickly test this code we provide a "dummmy" `infile.csv` file containing only a small number of records. **To repeat the analysis in the paper the dummy file provided here will need to be *replaced* using scripts and data in the `EnzymeTopt` folder.**

#### Usage
```
python run_model_evaluation.py infile.csv outdir
```

When run, the `run_model_evaluation.py` script will firstly standardize each column in the `infile.csv` file and then test the performance of six different regression models via a nested cross validation approach. Those six models are:

```python
sklearn.linear_model.LinearRegression
sklearn.linear_model.BayesianRidge
sklearn.linear_model.ElasticNetCV
sklearn.tree.DecisionTreeRegressor
sklearn.svm.SVR
sklearn.ensemble.RandomForestRegressor
```

`outdir` should be specified as the directory where output file will be created. The name of the outputfile is `infile.out`. It is a csv file contains the mean and standard deviation of R2 score in the cross validation, for example:
```
model,mean,std
Linear model,0.2288271120153847,0.06005441958706136
Elastic Net,0.10325699110912516,0.04805957255831517
BayesRige,0.22911720676693995,0.057324310305612164
SVR model,0.41176550406832035,0.030592246191507147
Tree model,0.14319793875641068,0.04974297243407614
Random forest,0.37394048308971517,0.033674531456666434
```
In which, mean and std are the average and standard deviation of R^2 scores from five-fold cross-validation.
#### Computation time
Depending on the number of features, the script would take minutes to several hours.
