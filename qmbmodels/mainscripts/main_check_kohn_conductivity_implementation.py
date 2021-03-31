#!/usr/bin/env python

"""
Module with functions used for calculating the Thouless
conductivity as defined in the 1972 paper by J. T. Edwards
and D. J. Thouless:
https://iopscience.iop.org/article/10.1088/0022-3719/5/8/007
However, we allow for complex hopping between all neighbouring
sites so that a phase factor is acquired upon each hopping. This
method is intended as a test of two implementations of the Kohn
conductivity.

The main idea is to calculate the spectrum of a (noninteracting)
Anderson Hamiltonian for a n-dimensional lattice first for
the model where all the hoppings are real except for the ones
amoung boundary sites along one axis, where some general
phase factor is obtaned. Then, we compare the results with
the case where the phase factor is distributed among all 
the hoppings along the specified direction.

This code is currently only intended for usage
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
    allowed_models = ['anderson_complex_hopping']
    if model_name not in allowed_models:
        print(('Thouless conductivity calculation '
               f'only works for the {allowed_models[0]} case '
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

        # get the unperturbed hamiltonian first
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization for the unperturbed pbc case...')
        eigvals_pbc = model.eigvals(complex=True)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the unperturbed pbc case finished!')

        # -------------------------------------------------------------
        #
        #
        # CHANGE OF THE BC ALONG ONE DIRECTION
        #
        # -------------------------------------------------------------

        # we also repeat the calculation for the case with
        # slightly changed bc. First, make sure
        # that boundary conditions are changed across one of the axes.
        argsDict_cplx_1 = argsDict.copy()

        # make sure the bc are of the proper shape
        argsDict_cplx_1['pbc'] = np.array(
            [1 for i in range(argsDict['dim'])], dtype=np.complex128)

        bc_modulus = argsDict_cplx_1['mod_bc']
        bc_phase = argsDict_cplx_1['phase_bc']
        argsDict_cplx_1['pbc'][-1] = bc_modulus * np.exp(1j * bc_phase)
        print('pbc and hopping amplitudes for case 1:')
        print(argsDict_cplx_1['pbc'])
        print(argsDict_cplx_1['t'])
        model, fields = mod.construct_hamiltonian(
            argsDict_cplx_1, parallel=False, mpisize=1)
        print('Starting diagonalization for the complex bc case...')
        eigvals_cplx_1 = model.eigvals(complex=True)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization for the complex bc case finished!')

        # case with complex hoppings and pbc
        argsDict_cplx_2 = argsDict.copy()
        hop_cplx_2 = argsDict_cplx_2['t']
        argsDict_cplx_2['t'] = np.array(
            [hop_cplx_2 for i in range(argsDict['dim'])], dtype=np.complex128)

        # each hopping along that direction gains a complex phase
        argsDict_cplx_2['t'][-1] *= np.exp(
            1j * argsDict_cplx_2['phase_bc'] / argsDict_cplx_2['L'])
        print('pbc and hopping amplitudes for case 2:')
        print(argsDict_cplx_2['pbc'])
        print(argsDict_cplx_2['t'])
        model, fields = mod.construct_hamiltonian(
            argsDict_cplx_2, parallel=False, mpisize=1)
        print('Starting diagonalization for the complex hopping case case...')
        eigvals_cplx_2 = model.eigvals(complex=True)

        # differences between spectra -> three types
        diffs_cplx_1 = eigvals_pbc - eigvals_cplx_1
        diffs_cplx_2 = eigvals_pbc - eigvals_cplx_2
        diffs_cplx_3 = eigvals_cplx_1 - eigvals_cplx_2
        print(('Displaying differences between the unperturbed '
               'and general complex bc spectra!'))
        print(diffs_cplx_1)
        print(('Displaying differences between the unperturbed '
               'and complex hoppings spectra!'))
        print(diffs_cplx_2)
        print(('Displaying differences for diffent '
               'introductions of the flux!'))
        print(diffs_cplx_3)
        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals_pbc,
                        f'Spectrum_phase_factor_{bc_phase:.8f}':
                        eigvals_cplx_1,
                        (f'Spectrum_phase_factor_'
                         f'complex_hopping_{bc_phase:.8f}'): eigvals_cplx_2,
                        ('Spectrum_differences_phase_factor'
                         f'_{bc_phase:.8f}'): diffs_cplx_1,
                        ('Spectrum_differences_2_phase_factor'
                         f'_{bc_phase:.8f}'): diffs_cplx_2,
                        ('Spectrum_differences_3_phase_factor'
                         f'_{bc_phase:.8f}'): diffs_cplx_3,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
