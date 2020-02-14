### Description:
* `MonteCarloSimulation` contains the script for Monte Carlo simulations in Figure 2.
  
* `EvaluateClassicalModels` contains the script to evaluate the performance of six different classical regression models on a given dataset. This script was used in the enzyme Topt prediction section.
  
* `EnzymeTopt` contains the two fasta files for enzyme Topt: ones with raw and cleaned Topt records. It also contains the script for feature extraction.
  
* `Genomics` contains the scripts, datasets and results for the evaluation of a random forest model on the prediction 34 quantitative phenotypes from pan genome. The pan genome data was taken from https://zenodo.org/record/3407352#.Xe-gX5P0nUI. Quantitative phenotypes were taken from *Peter, J. et al. Genome evolution across 1,011 Saccharomyces cerevisiae isolates. Nature 556, 339–344 (2018)*.
  
* `DeepNN` contains the architecture of deep neural networks used in the prediction of enzyme Topt.
  
* `Transcriptomics` contains scripts for the transcriptomics data analysis


### Python Packages required the versions used in the study
Python 3.6.7  
* numpy 1.15.0
* pandas 0.23.4
* sklearn 0.20.3
* seaborn 0.9.0
* keras 2.2.4 
* tensorflow 1.10.0
* hyperopt 0.1.2  
