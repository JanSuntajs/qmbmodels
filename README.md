# qmbmodels

This package contains functionalities for performing
exact and partial diagonalization studies on 1D quantum
hamiltonians as well as routines for running the code
on SLURM-based clusters. Apart from the diagonalization
routines the package also contains tools for calculating
various spectral statistics, such as the spectral form
factor, mean ratio of the adjacent level spacings, number
level variance and the entanglement entropy. The code works
in both spin and fermionic cases.

## Getting started

First clone or download this repository to some folder on your
machine. In order to get up-and-running, first run the
configuration file, which is, at the moment, just a simple
bash script:
```bash

#!/usr/bin/bash

conda deactivate
conda create --name petscenv --file conda_spec_file.txt
source activate petscenv

pip install . --process-dependency-links

```
```bash prepenv.sh ```

This should create an appropriate ```conda``` environment
as well as install all the user-defined package dependencies,
such as the [ham1d package](https://github.com/JanSuntajs/ham1d)
or the [spectral statistics package](https://github.com/JanSuntajs/spectral_statistics_tools).
To activate the environment, use the ```conda activate petscenv``` command.
Should you encounter any issues with the execution of the above
script, replacing the command ```source activate```
with ```conda activate``` should most likely fix the trouble.
