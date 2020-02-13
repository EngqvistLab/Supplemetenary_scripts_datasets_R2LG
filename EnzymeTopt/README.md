### There are two fasta files in this folder:
`all_enzyme_topts_v1.fasta` contains enzymes sequences with available optimal temperature in BRENDA.

In `cleaned_enzyme_topts_v1.fasta`, those Topts that are marked as 'assay at' in BRENDA were removed. Several other additional steps were applied to clean the Topt data. Check details in the manuscripts. 

The title line of each sequence is in the following format:
`>UniprotID ogt=54;topt=70`
in which ogt is the optimal growth temperature of the enzyme source organism and topt is the optimal functional temperature.

Incase of no OGT available, `nan` is used.
`>UniprotID ogt=nan;topt=45`

### Feature extraction with iFeature
This can be done with script `extract_iFeatures.sh`. 

### Extract UniRep encoding
This can be done with script `get_unirep_encoding.py`. For the script, one need to download UniRep Repo
```
git clone https://github.com/churchlab/UniRep.git
cd UniRep

# Then download 1900_weights
mv ../get_unirep_encoding.py .
```
