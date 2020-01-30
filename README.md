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

## First calculations

In order to run the calculations, it is the most convenient to first prepare a runner script, such as the one shown below, which is intended for running simulations of the Heisenberg "weak-link" model, which was used in the paper on
[Spin subdiffusion in the disordered Hubbard chain](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.246602) by M. Kozarzewski, P. Prelovšek and M. Mierzejewski (2018). The runner script is as follows:
```python
from senderscripts.batchsend import BatchSender
from utils.cmd_parser_tools import mode_parser

from models.prepare_model import import_model
import numpy as np

if __name__ == '__main__':
    storage = '.' # dictates where the results are stored
    ham_type = 'spin1d' # other options are: 'ferm1d', 'free1d', 'spin1d_kron'
    model = 'heisenberg_weak_links' # other currently implemented: 'imbrie', 'heisenberg'
    mod = import_model(model)
    params = {
        'L': [10], # system size
        'nu': [5], # number of up-spins
        'J': [1.], # maximum possible J
        'lambda': [1.], # parameter of the disorder distribution (see paper for more details)
        'W': [0.001], # random potential disorder
        'pbc': [True], # periodic boundary conditions
        'ham_type': [ham_type], # see above
        'disorder': ['powerlaw'], # choose the disorder distribution -> only powerlaw is allowed for this model
        'min_seed': [3], # minimum disorder seed -> relevant for different disorder realizations
        'max_seed': [7], # maximum disorder seed
        'step_seed': [2], # matters if job_type is sinvert_short only. Specifies how many different seeds are
                          # considered sequentially
        'sff_min_tau': [-5], # parameter for sff calculations -> minimum (unfolded) tau exponent -> 10 ** sff_min_tau
        'sff_max_tau': [2],  # maximum (unfolded) tau exponent -> 10 ** sff_max_tau
        'sff_n_tau': [30], # number of sff tau values
        'sff_eta': [0.5],  # eta for sff filtering
        'sff_unfolding_n': [3], # sff unfolding polynomial degree
        'sff_filter': ['gaussian'], # type of filtering to use in the sff calculation
        'r_step': [0.05], # in case <r> is calculated on the entire spectrum, this determines for how many 
                          # percentages of states the quantity is calculated
    }
  
    # Only relevant if shift-and invert method is used
    # shift and invert parameters -> leave this as it is for now except for '-eps_nev'
    sinvert_params = ['--model={}'.format(model),
                      '-eps_type krylovschur', '-eps_nev 10', # '-eps_nev' -> selects the number of eigenvalues
                      '-st_type sinvert',
                      '-st_ksp_type preonly',
                      '-st_pc_type lu',
                      '-st_pc_factor_mat_solver_type mumps',
                      '-mat_mumps_icntl_28 2',
                      '-mat_mumps_icntl_29 2']

    syspar_keys = mod.syspar_keys
    modpar_keys = mod.modpar_keys
    auxpar_keys = ['sff_min_tau', 'sff_max_tau', 'sff_n_tau', 'sff_eta',
                   'sff_unfolding_n', 'sff_filter', 'r_step']
    
    # select a human readable name for the job
    name = f'{model}_{ham_type}_test'
    # these parameters are only relevant if job is ran on the cluster
    time = "00:59:59"
    nodes = 1   # number of nodes
    ntasks = 1  # number of threads
    memcpu = 4  # memory in GB per CPU!
    cputask = 2

    # this takes care of the job's execution
    queue, mode = mode_parser()

    print(f"{mode}")

    sender = BatchSender(params, syspar_keys, modpar_keys,
                         auxpar_keys, cmd_opt=sinvert_params,
                         storage=storage)

    sender.run_jobs(mode, queue=queue,
                    time=time, nodes=nodes, ntasks=ntasks,
                    memcpu=memcpu, name=name, cputask=cputask,
                    sourcename='petscenv')


```
## Running the code
To run an example script ```runner_script.py``` on your home machine and calculate various
quantities of interest, do the following:

```python runner_script.py diag sff gaps```

This would run the full diagonalization script, then calculate the spectral form factor and finaly the mean
ratio of the adjacent level spacing. As long as the diagonalization data already exist, one can calculate
sff and the mean ratio independently as well. The above command would run the code on a home machine or on
the cluster's headnode. To send the jobs to SLURM, do the following:

```python runner_script.py --queue diag sff gaps```

To perform a partial diagonalization calculation instead of a full one, do one of the following:

```python runner_script.py --queue sinvert sff gaps_partial```

```python runner_script.py --queue sinvert_short sff gaps_partial```

Note how ```gaps``` was replaced with ```gaps_partial``` as well.
