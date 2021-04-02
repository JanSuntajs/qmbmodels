#!/usr/bin/env python

"""
Module with functions used for calculating the Thouless
conductivity as defined in the 1972 paper by J. T. Edwards
and D. J. Thouless:
https://iopscience.iop.org/article/10.1088/0022-3719/5/8/007

The main idea is to calculate the spectrum of a interacting
XXZ Heisenberg Hamiltonian in one dimension. We calculate
the energy spectrum first for periodic boundary conditions
then change that to antiperiodic and calculate the difference.

This module currently repeats lots of the code used
in the non-interacting case, however, to avoid
confusion, we use a separate module for now.

"""
import sys
import numpy as np

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

save_metadata = True

if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    # make sure that the input parameters are ok before
    # proceeding -> the hamiltonian should be of the Anderson
    # type and the selected boundary conditions should be
    # periodic along all the axes -> the pbc parameter is
    # simply a scalar boolean with the value of True
    if model_name != 'heisenberg_complex':
        print(('Thouless conductivity calculation '
               'for now only works for the '
               'heisenberg_complex module '
               'in the interacting case. '
               f'{model_name} is not supported. Exiting.'))
        sys.exit()

    if argsDict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsDict["pbc"]} which is not '
               'supported. Exiting.'))
        sys.exit()
    # to avoid confusion, make sure there are no
    # complex phases present
    if ((argsDict['J_phase'] != 0.0) and (argsDict['phase_bc'] != 0.0)):
        print(('Please make sure all the complex '
               'phase factors are set to zero!'
               ' Exiting.'))
        sys.exit()

    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization for the pbc case...')
        eigvals_pbc = model.eigvals(complex=False)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the pbc case finished!')

        # repeat the calculation for the apbc case.
        argsDict_apbc = argsDict.copy()
        # make sure the bc are of the proper shape
        argsDict_apbc['phase_bc'] = np.pi

        model, fields = mod.construct_hamiltonian(
            argsDict_apbc, parallel=False, mpisize=1)
        print('Starting diagonalization for the apbc case...')
        eigvals_apbc = model.eigvals(complex=False)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the apbc case finished!')

        spectrum_differences = eigvals_pbc - eigvals_apbc
        print('Displaying differences between spectra:')
        print(spectrum_differences)
        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals_pbc,
                        'Spectrum_apbc': eigvals_apbc,
                        'Spectrum_differences': spectrum_differences,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
