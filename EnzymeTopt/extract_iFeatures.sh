# Firstly download iFeature from Github
git clone https://github.com/Superzchen/iFeature.git

mkdir Extracted_iFeatures_all_enzymes
mkdir Extracted_iFeatures

cd iFeature

for tpe in  {AAC,,CKSAAP,DPC,DDE,GAAC,CKSAAGP,GDPC,GTPC,NMBroto,Moran,Geary,CTDC,CTDT,CTDD,CTriad,KSCTriad,SOCNumber,QSOrder,PAAC,APAAC}
do python iFeature.py --type $tpe --out ../Extracted_iFeatures_all_enzymes/$tpe.csv --file ../all_enzyme_topts_v1.fasta
done


for tpe in  {AAC,,CKSAAP,DPC,DDE,GAAC,CKSAAGP,GDPC,GTPC,NMBroto,Moran,Geary,CTDC,CTDT,CTDD,CTriad,KSCTriad,SOCNumber,QSOrder,PAAC,APAAC}
do python iFeature.py --type $tpe --out ../Extracted_iFeatures/$tpe.csv --file ../cleaned_enzyme_topts_v1.fasta
done
