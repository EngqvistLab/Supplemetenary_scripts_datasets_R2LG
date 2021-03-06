# Performance of regression models as a function of experiment noise
This repository contains code and data (or link to needed data) needed to replicate the analysis carried out in the pre-print by  Li G, et al, 2019 (https://arxiv.org/abs/1912.08141).

### Description
The different parts of the analysis is broken down into separate folders:

* `MonteCarloSimulation` contains the script for Monte Carlo simulations in Figure 2.

* `EvaluateClassicalModels` contains the script to evaluate the performance of six different classical regression models on a given dataset. This script was used in the enzyme Topt prediction section (Figure 3).

* `EnzymeTopt` contains the two fasta files for enzyme Topt: ones with raw and cleaned Topt records. It also contains the script for feature extraction (Figure 3).

* `DeepNN` contains the architecture of deep neural networks used in the prediction of enzyme Topt (Figure 3).

* `Transcriptomics` contains scripts for the transcriptomics data analysis (Figure 4).

* `Genomics` contains the scripts, datasets and results for the evaluation of a random forest model on the prediction 34 quantitative phenotypes from pan genome (Figure 5). The pan genome data was obtained from https://zenodo.org/record/3407352#.Xe-gX5P0nUI. Quantitative phenotypes were obtained from *Peter, J. et al. Genome evolution across 1,011 Saccharomyces cerevisiae isolates. Nature 556, 339–344 (2018)*.



### Dependencies
```
numpy 1.15.0
pandas 0.23.4
sklearn 0.20.3
seaborn 0.9.0
keras 2.2.4
tensorflow 1.10.0
hyperopt 0.1.2  
```
The repository was tested with Python 3.6.7.

### Hardware
Scripts of `MonteCarloSimulation` and `Transcriptomics` can be performed with PC with linux or Mac OS. Other scripts need to be done with a computer cluster. All deep learning related analysis (UniRep representation and `DeepNN`) need GPU platform.
