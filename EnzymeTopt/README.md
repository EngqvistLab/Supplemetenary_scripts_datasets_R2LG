### Description
This folder contains all data needed to carry out the Topt analysis (Figure 3). The protein sequences were either used for feature-extraction using IFeature, for extracting embeddings with UniRep, or directly for training a deep NN.

#### Datasets
`all_enzyme_topts_v1.fasta` contains enzymes sequences with available optimal temperature in BRENDA (version 2019.1).

In `cleaned_enzyme_topts_v1.fasta`, those Topts that are marked as 'assay at' in BRENDA were removed as they are not likely to represent true catalytic optima. Several other additional steps were applied to clean the Topt data. Check details in the manuscripts.

The title line of each sequence is in the following format:
`>UniprotID ogt=54;topt=70`
in which ogt is the optimal growth temperature of the enzyme source organism and topt is the optimal functional temperature.

In case that no OGT is available, `nan` is used.
`>UniprotID ogt=nan;topt=45`

#### Feature extraction with iFeature
iFeature features can be obtained with the script `extract_iFeatures.sh`. The script automatically downloads and runs code from the iFeatures repository https://github.com/Superzchen/iFeature.git.

#### Extract UniRep encoding
UniRep embeddings can be obtained by running the script `get_unirep_encoding.py`. For the script, one need to first download UniRep Repository and 1900_weights according to the instructions in UniRep repository.
```
git clone https://github.com/churchlab/UniRep.git
cd UniRep
mv ../get_unirep_encoding.py .
python get_unirep_encoding.py ../all_enzyme_topts_v1.fasta
```
