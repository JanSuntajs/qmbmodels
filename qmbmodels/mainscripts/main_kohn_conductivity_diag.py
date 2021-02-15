#!/usr/bin/env python

"""
Module with functions used for calculating the Thouless
conductivity as defined in the 1972 paper by J. T. Edwards
and D. J. Thouless:
https://iopscience.iop.org/article/10.1088/0022-3719/5/8/007
However, we allow for a general complex phase factor to be obtained
once a particle traverses the system boundary. Consequently,
we generally have to deal with complex matrices here.

The main idea is to calculate the spectrum of a (noninteracting)
Anderson Hamiltonian for a n-dimensional lattice first for
periodic boundary conditions along all the axes and then
changing the boundary conditions along one of the axes
so that a phase factor of np.exp(1j*alpha) is obtained
when traversing the boundary.
This code is only intended for usage
with non-interacting Hamiltonians.

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

    # we also repeat the calculation for the case with
    # slightly changed bc. First, make sure
    # that boundary conditions are changed across one of the axes.
    argsDict_apbc = argsDict.copy()
    # make sure that the input parameters are ok before
    # proceeding -> the hamiltonian should be of the Anderson
    # type and the selected boundary conditions should be
    # periodic along all the axes -> the pbc parameter is
    # simply a scalar boolean with the value of True
    if model_name != 'anderson_complex':
        print(('Thouless conductivity calculation '
               'only works for the Anderson case. '
               f'{model_name} is not supported. Exiting.'))
        sys.exit()

    if argsDict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsDict["pbc"]} which is not '
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

        # make sure the bc are of the proper shape
        argsDict_apbc['pbc'] = np.array(
            [1 for i in range(argsDict['dim'])], dtype=np.int32)

        ph_factor = argsDict_apbc['boundary_phase']
        argsDict_apbc['pbc'][-1] = ph_factor
        model, fields = mod.construct_hamiltonian(
            argsDict_apbc, parallel=False, mpisize=1, dtype=np.complex128)
        print('Starting diagonalization for the general bc case...')
        eigvals_apbc = model.eigvals(complex=True)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the general bc case finished!')

        spectrum_differences = eigvals_pbc - eigvals_apbc
        print('Displaying differences between spectra:')
        print(spectrum_differences)
        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals_pbc,
                        f('Spectrum_phase_factor_{ph_factor.real:.2f}'
                          f'_{ph_factor.imag:.2f}j'): eigvals_apbc,
                        ('Spectrum_differences_phase_factor'
                         f'_{ph_factor.real:.2f}_{ph_factor.imag:.2f}j'): spectrum_differences,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
