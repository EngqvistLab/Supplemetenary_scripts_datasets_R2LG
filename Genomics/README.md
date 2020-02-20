### Description
This folder contains all code and data required for the *S. cerevisiae* genomics analysis (Figure 5).

#### Datasets
All input data is available in the `data` folder.

#### Usage
`machine_learning_CNV_on_phenome.py`: Evaluates the performance of a random forest regressor on predicting phenome from gene copy number variation table.


`machine_learning_PA_on_phenome.py`: Evaluates the performance of a random forest regressor on predicting phenome from gene presence/absecne table.

The results from above two scripts can be visualized with the jupyter notebook `visualize_results_PA_CNA_on_35_phenotypes.ipynb`
