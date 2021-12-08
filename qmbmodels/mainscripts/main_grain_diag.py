#!/usr/bin/env python

"""
Diagonalization routine especially for the grain
model, featuring tools for the analysis of the matrix
elements. We focus on the grain model similar to the
one used in the following paper:
https://arxiv.org/abs/1705.10807
(How a small quantum bath can thermalize long ergodic chains by
David J. Luitz, Francois Huveneers, Wojciech de Roeck).

First, we perform the full numerical diagonalization of the model
Hamiltonian to obtain both eigenvalues and their corresponding
eigenstates. Afterwards, we evaluate the matrix elements, 
first for the farthest LIOM in the localized regime, then also
for the sum of all the LIOMs in the localized regime.


We calculate:

The ratios of variances of diagonal and offdiagonal
matrix elements. See here (around eq. 27)) for implementational
details: https://arxiv.org/pdf/1902.03247.pdf
(paper titled Eigenstate Thermalization and quantum chaos in the Holstein
polaron model by David Janssen, Jan Stolpp, Lev Vidmar and Fabian Heidrich
Meissner). For testing purposes, we allow for choices of different widths
of the microcanonical windows in the same calculation (since the diagonalization
and matrix elements calculation are the most computationally demanding tasks
and everything else is just postprocessing.) In this calculation, we also extract
the mean energies in the chosen microcanonical windows and the variances of both
diagonal and offdiagonal matrix elements in the selected microcanonical windows.

Susceptibilities


Spectral functions and integrated spectral functions
"""
import numpy as np
import sys
import gc

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from _spectral_fun import eval_spectral_fun, eval_matelt_variances, calc_susceptibility, _smoothing, sort_offdiag_matelts

save_metadata = True
_allowed_models = ['rnd_grain']


_spc_parse_dict = {
    # relative to the hilbert space dimension
    'matelt_variance_window_min': [float, 1.],
    # relative to the hilbert space dimensioon
    'matelt_variance_window_max': [float, 1.],
    # relative to the hilbert space dimension
    'matelt_variance_window_nsteps': [int, 1],
    # absolute terms - number of terms
    'full_spectral_smoothing_window_min': [int, 1],
    'full_spectral_smoothing_window_max': [int, 1],  # to consider in smoothing
    'full_spectral_smoothing_window_nsteps': [int, 1],  # -II-
    'partial_spectral_smoothing_window_min': [int, 1],  # -II-
    'partial_spectral_smoothing_window_max': [int, 1],  # -II-
    'partial_spectral_smoothing_window_nsteps': [int, 1],  # -II-
    # relative to the bandwidth
    'partial_spectral_eps_min': [float, 0.01],
    # relative to the bandwidth
    'partial_spectral_eps_max': [float, 0.01],
    'partial_spectral_eps_nsteps': [int, 1]

}

# for storing the susceptibility, matrix elements variances
# and spectral function results

# first key is for the window width, the second for the relative window width, the third for the operator type
_mateltkeys = ['microcan_energies_window_{:d}_{:.5f}_{}',
               'diagonal_variances_window_{:d}_{:.5f}_{}',
               'offdiagonal_variances_window_{:d}_{:.5f}_{}',
               'variance_ratios_window_{:d}_{:.5f}_{}']

_susckeys = ['susceptibilities_nofilter_{}',
             'log_susceptibilities_nofilter_{}',
             'susceptibilities_filter_{}',
             'log_susceptibilities_filter_{}',
             'mu_susceptibility_filter_value_{},'
             ]

_spectralkeys = ['diagonal_matelts_{}',
                 'omega_full_spectrum_smoothing_{:d}_{}',
                 'spectral_function_full_smoothing_{:d}_{}',
                 'spectral_integrated_function_full_smoothing_{:d}_{}',
                 'omega_partial_spectrum_smoothing_{:d}_eps_{:.5f}_{}',
                 'spectral_function_partial_smoothing_{:d}_eps_{:.5f}_{}']


if __name__ == '__main__':

    spcDict, spcextra = arg_parser_general(_spc_parse_dict)

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    # sanity check
    if model_name not in _allowed_models:
        print(('Random grain analysis '
               f'only works for the {_allowed_models[0]} case '
               f'{model_name} is not supported. Exiting.'))
        sys.exit()

    for seed in range(min_seed, max_seed):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        # set complex = False to speed up things
        print('Starting diagonalization!')
        eigvals, eigvecs = model.eigsystem(complex=False)
        print('Diagonalization finished!')
        # mean level spacing:
        mn_lvl_spc = np.mean(np.diff(eigvals))

        # -----------------------------------------------------------------
        #
        #  MATELTS CALCULATION - for the local and "global" sz operator
        #  Local Sz: S^z_L
        #  Global: normalized sum over the LIOMs
        # -----------------------------------------------------------------
        loc_op_desc = ['z', [[1, model.L - 1]]]

        lmin = argsDict['L_b']
        lloc = model.L - lmin
        norm_factor = 1. / np.sqrt(lloc)
        glob_op_desc = ['z', [[norm_factor, i] for i in range(lmin, model.L)]]

        oplist = [loc_op_desc, glob_op_desc]
        namelist = ['local_sz', 'global_sz']
        matelts_dict = {}
        # eval matelts -> first the local term: the most distant LIOM

        for i, op_desc in enumerate(oplist):
            print(f'Operator {op_desc} matrix elements calculations starting!')

            print('Starting matelts calculation!')
            matelts = model.eval_matelts(eigvecs, op_desc, dtype=np.float64)
            print('Matelts calculation finished!')
            # ---------------------------------------------------------------
            #
            # Variances
            #
            # ---------------------------------------------------------------

            # evaluate variances of the matrix elements

            # analyze the ratios of variances of the matrix elements first
            # mc_energies: mean energies in the microcanonical windows
            # variances_diag: variances of the diagonal matrix elements in selected energy
            # windows
            # variances_offdiag: variances of the offdiagonal matrix elements
            # ratios: ratios of diagonal and offdiagonal matrix elements variances
            print('Starting matrix variance calculations!')
            for j in np.linspace(spcDict['matelt_variance_window_min'],
                                 spcDict['matelt_variance_window_max'],
                                 spcDict['matelt_variance_window_nsteps'], dtype=float):

                window_width = int(j * eigvals.size)
                results = eval_matelt_variances(
                    eigvals, matelts, window_width)

                for k in range(len(results)):

                    matelts_dict[_mateltkeys[k].format(
                        window_width, j, namelist[i])] = results[k].copy()

                print(f'Matrix calculations, calculation for window {window_width} finished!')
            print('Matrix variance calculations finished!')

            results.clear()
            gc.collect()
 
            # ----------------------------------------------------------------
            #
            # SUSCEPTIBILITIES
            #
            # ----------------------------------------------------------------
            print('Starting susceptibility calculations!')
            mu = mn_lvl_spc * model.L
            results = calc_susceptibility(eigvals, matelts, mu )
            for j in range(len(results)):
                if j <= 1:
                    mu_ = 0.0
                else:
                    mu_ = mu
                matelts_dict[_susckeys[j].format(namelist[i])] = results[j].copy()

            print('Susceptibility calculations finished!')

            results.clear()
            gc.collect()
            # ----------------------------------------------------------------
            #
            # Full matrix elements distribution
            #
            # ----------------------------------------------------------------
            # now, perform the analysis -> first the full distribution of
            # matelts, then from some energy window; also calculate
            # the susceptibilities
            print('Full spectrum spectral function calculation starting!')
            diags, diffs, matelts, aves = sort_offdiag_matelts(eigvals, matelts)
            print('Sorted offdiagonals!')
            diffs_full, spc_fun_full, spc_fun_integ, *_ = eval_spectral_fun(
                eigvals, aves, diffs, matelts, 0., 0., True,
            )
            print('Full spectrum spectral function calculation finished! Starting smoothing!')
            matelts_dict[_spectralkeys[0].format(namelist[i])] = diags.copy()

            for j in np.linspace(spcDict['full_spectral_smoothing_window_min'],
                                 spcDict['full_spectral_smoothing_window_max'],
                                 spcDict['full_spectral_smoothing_window_nsteps'], dtype=int):
                # smoothing of the results

                diffs_, spc_ = _smoothing(diffs_full, spc_fun_full, j)
                *_, spc_integ_ = _smoothing(diffs_full, spc_fun_integ, j)

                results = (diffs_, spc_, spc_integ_)

                for k in range(len(results)):
                    matelts_dict[_spectralkeys[k+1].format(
                        j, namelist[i])] = results[k].copy()

            print('Smoothing for the full spectrum spectral function finished!')
            results.clear()
            gc.collect()
            # -----------------------------------------------------------------
            #
            # Matrix elements inside a window
            #
            # -----------------------------------------------------------------
            # now the window -> do not calculate susceptibilities,
            # integrated function or perform any calculations for the full spectrum
            # concentrate on the window around the mean energy
            # allow for testing both eps and smoothing values independently of
            # the full distribution
            print('Performing the partial spectral function calculations!')
            for eps in np.linspace(spcDict['partial_spectral_eps_min'],
                                   spcDict['partial_spectral_eps_max'],
                                   spcDict['partial_spectral_eps_nsteps'], dtype=float):

                diffs_, spc_fun_, _, n_vals, target_ene, eps_, bandwidth = eval_spectral_fun(
                    eigvals, aves, diffs, matelts, np.mean(eigvals), eps, False,)

                for j in np.linspace(spcDict['partial_spectral_smoothing_window_min'],
                                     spcDict['partial_spectral_smoothing_window_min'],
                                     spcDict['partial_spectral_smoothing_window_min'], dtype=int):

                    results = _smoothing(diffs_, spc_fun_, j)

                    for k in range(len(results)):
                        matelts_dict[_spectralkeys[k +
                                                    4].format(j, eps, namelist[i])] = results[k].copy()

        print('Finished performing the partial spectral function calculations!')
        results.clear()
        gc.collect()
        # ---------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals,
                        **matelts_dict,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
