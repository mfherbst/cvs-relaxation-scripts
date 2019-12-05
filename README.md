# Quantifying the error of the core-valence separation approximation
[![](https://img.shields.io/badge/arxiv-2005.05848-red)](https://arxiv.org/abs/2005.05848)

Implementation of the Rayleigh-Quotient-based CVS relaxation discussed in  

Michael F. Herbst, Thomas Fransson  
*Quantifying the error of the core-valence separation approximation*  
Preprint on [arxiv](https://arxiv.org/abs/2005.05848)

This repository contains the relaxation code, data and example scripts
for reproducing the results discussed in the paper.
The implementation makes use of [adcc](https://adc-connect.org) version 0.14.2.
For more details see also [this blog article](https://michael-herbst.com/2020-cvs-relaxation.html).

## Table of contents
- [relaxation.py](relaxation.py): Script containing the QR-based CVS relaxation procedure
- [ammonia.py](ammonia.py), [fluoro_ethene.py](fluoro_ethene.py), [methane.py](methane.py),
  [water.py](water.py), [water_singles.py](water_singles.py): A few examples how to
  run CVS relaxations. See below for details how to install requirements and run these scripts.
- [basis](basis): Basis set definitions for the modified `6-311++G**` basis sets used
  in the study (`p6-311++G**`, `u6-311++G**`, `up6-311++G**`)
- [geometries](geometries): `xyz`-files containing the MP2-optimised geometries,
  which were used as starting points. Length unit in the files is Ångström.
- [data](data): Various files containing raw data for figures and tables.

## Running the code
Required is at least **python 3.6**.
The installation process further requires
`pipenv` ([Installing pipenv](https://pipenv.pypa.io/en/latest/install/#installing-pipenv)).  

To install all dependencies and run the example scripts just type:
```
pipenv install  # Install dependencies into separate python environment
pipenv shell    # Get a shell in this environment

./water.py      # Run example scripts
```
