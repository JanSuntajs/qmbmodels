#!/usr/bin/env python

"""
Module with functions used for calculating the Thouless
conductivity as defined in the 1972 paper by J. T. Edwards
and D. J. Thouless:
https://iopscience.iop.org/article/10.1088/0022-3719/5/8/007

The main idea is to calculate the spectrum of a (noninteracting)
Anderson Hamiltonian for a n-dimensional lattice first for
periodic boundary conditions along all the axes and then
changing the boundary conditions along one of the axes
to anti-periodic ones. The difference of spectra is then
calculated.
This code is only intended for usage
with non-interacting Hamiltonians.

"""
import sys

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
    if model_name != 'Anderson':
        print(('Thouless conductivity calculation '
               'only works for the Anderson case. '
               f'{model_name} is not supported. Exiting.'))
        sys.exit()

    if argsDict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsDict['pbc']} which is not '
               'supported. Exiting.'))
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

        print('Displaying pbc eigvals')
        print(eigvals_pbc)

        # repeat the calculation for the apbc case. First, make sure
        # that boundary conditions are changed across one of the axes.

        # make sure the bc are of the proper shape
        argsDict['pbc'] = np.array(
            [1 for i in range(argsDict['dim'])], dtype=np.int32)
        # choose antiperiodic bc along the last axis
        argsDict['pbc'][-1] = -1

        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)
        print('Starting diagonalization for the apbc case...')
        eigvals_apbc = model.eigvals(complex=False)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the apbc case finished!')

        print('Displaying apbc eigvals')
        print(eigvals_apbc)

        spectrum_differences = eigvals_pbc - eigvals_apbc

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
