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

    # make sure that the input parameters are ok before
    # proceeding -> the hamiltonian should be of the Anderson
    # type and the selected boundary conditions should be
    # periodic along all the axes -> the pbc parameter is
    # simply a scalar boolean with the value of True

    if 'complex' not in model_name:
        if 'anderson' in model_name:

            print(('Calculation only works for the '
                   'anderson_complex module in the '
                   'anderson case. '
                   f'{model_name} is not supported. Exiting.'))
        if 'heisenberg' in model_name:

            print(('Calculation only works for the '
                   'heisenberg_complex module in the '
                   'heisenberg case. '
                   f'{model_name} is not supported. Exiting.'))
        sys.exit()

    # if we are not dealing with an anderson-type model
    # then this is an interacting case in 1D!
    interacting = False
    if 'anderson' not in model_name:
        interacting = True

    if interacting:
        # make sure not both types of complex factors
        # are nonzero
        if ((argsDict['J_phase'] != 0.) and (argsDict['phase_bc'] != 0.0)):
            print(('Please make sure not both phase factors are nonzero. '
                   'Exiting.'))
            sys.exit()
    if argsDict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsDict["pbc"]} which is not '
               'supported. Exiting.'))
        sys.exit()

    for seed in range(min_seed, max_seed + 1):
        # perform the first part of the calculation
        # as is
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed

        if interacting:
            # set the complex phases first to zero
            bc_phase_int = argsDict['phase_bc']
            hop_phase_int = argsDict['J_phase']
            argsDict['phase_bc'] = 0.
            argsDict['J_phase'] = 0.
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization for the pbc case...')
        eigvals_pbc = model.eigvals(complex=True)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the pbc case finished!')

        # we also repeat the calculation for the case with
        # slightly changed bc. First, make sure
        # that boundary conditions are changed across one of the axes.
        argsDict_apbc = argsDict.copy()

        # make sure the bc are of the proper shape
        # for the noninteracting case, pbc is an array,
        # because there are generally more dimensions to
        # consider

        savestr = ''
        if not interacting:
            argsDict_apbc['pbc'] = np.array(
                [1 for i in range(argsDict['dim'])], dtype=np.complex128)

            bc_modulus = argsDict_apbc['mod_bc']
            bc_phase = argsDict_apbc['phase_bc']
            argsDict_apbc['pbc'][-1] = bc_modulus * np.exp(1j * bc_phase)

        else:
            # note: we set the bc phase to 0 for the
            # original case, now we restore it; we do not
            # change the 'pbc' parameter since its implementation
            # is different than in the anderson case
            argsDict_apbc['J_phase'] = hop_phase_int
            argsDict_apbc['phase_bc'] = bc_phase_int
            argsDict['phase_bc'] = bc_phase_int
            argsDict['J_phase'] = hop_phase_int
            bc_phase = bc_phase_int
            if hop_phase_int != 0.:
                savestr = 'global'

        model, fields = mod.construct_hamiltonian(
            argsDict_apbc, parallel=False, mpisize=1)
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
                        f'Spectrum_{savestr}_phase_factor_{bc_phase:.8f}':
                        eigvals_apbc,
                        (f'Spectrum_differences_{savestr}_phase_factor'
                         f'_{bc_phase:.8f}'): spectrum_differences,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
