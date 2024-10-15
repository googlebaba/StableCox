[![DOI](https://zenodo.org/badge/864352564.svg)](https://doi.org/10.5281/zenodo.13852489)

# System Requirements
## Hardware requirements
`Stable Cox' package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS requirements
This package is supported for *Linux*. The package has been tested on the following system:
+ Linux: Ubuntu 18.04

### Python Dependencies
'Stable Cox' mainly depends on the Python scientific stack.

```
lifelines=0.27.8
numpy=1.20.3
pandas=2.0.3
scikit-learn=1.3.0
```

# Installation Guide
conda create -n Stable_Cox python=3.8

source activate  Stable_Cox

pip install -r requirements.txt

- This takes several mins to build

# Run demo

## omics data

### Stable Cox
python3 mRNA_HCC_OS.py --reweighting SRDO --paradigm regr --topN 10 
### Cox PH
python3 mRNA_HCC_OS.py --reweighting None --paradigm regr --topN 10

## clinical data

### Stable Cox
python3 clinical_lung_OS.py --reweighting SRDO --paradigm regr
### Cox PH
python3 clinical_lung_OS.py --reweighting None --paradigm regr


## simulated data

### Stable Cox
python3 simulated.py --reweighting SRDO --paradigm regr
### Cox PH
python3 simulated.py --reweighting None --paradigm regr


## feature selection
### Stable Cox
python3 simulated_fs.py --reweighting SRDO --paradigm regr --topN 5
### Cox PH
python3 simulated_fs.py --reweighting None --paradigm regr --topN 5

- The expected running time is from several seconds to mins depends on the number of samples.

# License
This project is licensed under the terms of the MIT license.
