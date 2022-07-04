#!/usr/bin/env python

"""
This module provides tools for performing a diagonalization
task and calculating the entanglement entropy for different subsystem
fractions of the Anderson model.

The entanglement entropy results are prepared for the following
subsystem ratios f=V_A / V; where V_A is the volume of the subsystem
and V the total system volume.

np.arange(0.1, 1.05, 0.05)
"""

import numpy as np
from time import time

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from _anderson_ententro_optimise import main_fun_entro
from qmbmodels.utils.cmd_parser_tools import arg_parser_general

try:

    print(('main_anderson_entro_diag info '
           'on the number of cores:'))
    set_mkl_lib.mkl_get_max_threads()
    # save the variable
    omp_ncores = set_mkl_lib.mkl_rt.MKL_Get_Max_Threads()

except NameError:
    print(('There was an error importing mkl libraries! '
           'Check numpy and scipy installation!'))
    omp_ncores = 1


try:
    from numba import get_num_threads, set_num_threads

    numba_ncores = get_num_threads()
    print(('Numba features for thread masking imported!'
           f'Number of numba threads is {get_num_threads()}'))

except ImportError:
    print(('Regular, not dev version of numba is used! '
           'Some advanced features might be missing!'))
    numba_ncores = 1

_eentro_parse_dict = {'eentro_nstates': [int, -1],
                      'eentro_filling': [float, 0.5],
                      'eentro_partition': [float, 0.5],
                      'eentro_grandcanonical': [int, 1]}

save_metadata = True

# this program relies on numba, but the main
# scripts set the number of cores via the
# OMP_NUM_THREADS environment variable


if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    # Obtain the command-line arguments relevant
    # to the entanglement entropy calculation:

    eentroDict, eentro_extra = arg_parser_general(_eentro_parse_dict)
    # number of many-body states for which we calculate
    # the entropy
    eentro_nstates = eentroDict['eentro_nstates']
    # filling fraction - ratio of occupied states
    # compared to the volume of the system
    filling = eentroDict['eentro_filling']

    partition_fraction = eentroDict['eentro_partition']

    gc = eentroDict['eentro_grandcanonical']

    for seed in range(min_seed, max_seed):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        print(argsDict)

        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization and entanglement calculation')
        # omp mode for diagonalization

        set_mkl_lib.mkl_set_num_threads(omp_ncores)
        set_num_threads(1)
        start = time()
        eigvals, eigvecs = model.eigsystem(complex=False)
        end = time()
        print(f'Elapsed diagonalization time: {end-start}')
        # now, in the numba part, set up for the numba parallelism
        # set_mkl_lib.mkl_set_num_threads(1)
        # set_num_threads(numba_ncores)

        eentro_results = {}
        for partition_frac in np.arange(0.1, .55, 0.05):
            entanglement_name = (f'Entropy_noninteracting_'
                                f'nstates_{eentro_nstates}_'
                                f'partition_size_{partition_frac:.2f}_'
                                f'filling_{filling:.2f}')
            start = time()
            eentro = main_fun_entro(eigvecs, model.states, eentro_nstates,
                                    partition_frac,
                                    filling, np.bool(gc))
            end = time()
            print(f'Elapsed eentro calculation time: {end-start}')

            eentro_results[entanglement_name] = eentro

        # ----------------------------------------------------------------------
        # save the files

        # do not save field configurations
        eigvals_dict = {'Eigenvalues': eigvals,
                        **eentro_results}

        argsDict.update(eentroDict)
        syspar_keys += list(eentroDict.keys())
        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
