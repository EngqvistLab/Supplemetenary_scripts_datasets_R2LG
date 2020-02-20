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
Linear model,0.5606470243036312,0.05182762180825625
Elastic Net,0.5561527172818607,0.046980868519725986
BayesRige,0.5613494374587507,0.051039246940720234
SVR model,0.547889662217659,0.026893484452659596
Tree model,0.5355491267446859,0.03453184980344376
Random forest,0.6021454315518417,0.028483260693779797
```

#### Computation time
Depending on the number of features, the script would take minutes to several hours.
