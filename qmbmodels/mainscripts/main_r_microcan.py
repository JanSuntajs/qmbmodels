#!/usr/bin/env python

"""
This module provides utilities for calculating
the mean ratio of the adjacent level spacings in
the microcanonical version. The calculation allows
for calculation of r for different temperatures in
energy windows of selected numbers of states centered
at mean energies corresponding to appropriate energies.

This implementation uses tools from the
spectral_statistics_tools package
available at:

https://github.com/JanSuntajs/spectral_statistics_tools


"""

import os
import numpy as np
import glob
import h5py

from spectral_stats.spectra import Spectra

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general

_r_keys = ['r_step']

_r_parse_dict = {'r_step': [float, 0.25], }

_r_name = 'r_data'
# sfflist text descriptor
header = """
This file provides data on the microcanonical analysis of the
average ratio of the adjacent level spacings (r). Instead
of calculating the ratio only for eigenstates within a narrow
energy window at the center of the spectrum, we do so for energy
windows corresponding to different temperatures. See footer for
more details on the contents of the file.

Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n
"""

footer = """
In this footer, we provide a description on the contents of the
columns in the file. Note that we typically calculate our microcanonical
averages such that the energy window is centered at the selected target
energy. This is typically not the case at the central edges where we take
the n_window most extreme values where n_window is the number of selected
energies in the window.
The columns in this file are ordered as follows:

0) Temperature
1) Mean target energy - calculated as a mean over different
   disorder realizations. For a given temperature, we calculate
   the target energy as a (grand)canonical average energy at the
   selected temperature.
2) Mean energy - (the microcanonical average) of all the energies
   in the selected window for which the average r is calculated.
   In most cases, 2) and 3) should be close, but discrepancies can
   occur, for instance, at the spectral edges due to the implementation
   of the calculation.
3) mean r - average level spacing ratio for the selected energy window.
4) std r - standard deviation of the r values in the window.
5) n_window - number of energies in the window
6) nener - number of all the energies
7) nsamples - number of random samples (disorder realizations)
8) Global mean energy - mean energy of all the spectra
9) Energy variance across all the spectra
10) Mean of the variances of individual spectra
11) Mean minimum energy of the whole spectra
12) Mean maximum energy of the whole spectra
13) Average median energy.

"""

def _set_target_ene(temperature, data):
    """
    An internal helper routine for setting the target
    energy of the spectrum corresponding to a given
    temperature.

    Parameters:

    temperature: float
    Sets the temperature of the (grand)canonical ansamble.

    data: array, 2D
    Energy spectra. Should be of the shape (nsamples, nener)
    where nsamples is the number of the random samples and
    nener is the number of calculated energies.

    Returns:

    target_ene: array, 1D
    Array of target energies.

    target_idx: array, 1D
    Array of first indices for which the data are larger or equal to
    the target energy (for each sample).
    """

    target_ene = np.sum(data * np.exp(-data / temperature),
                        axis=1) / np.sum(np.exp(-data/temperature), axis=1)
    target_idx = np.argmax(data >= target_ene[:, np.newaxis], axis=1)
    return target_ene, target_idx


def _set_window(window, nener, target_idx, i):

    lohalf = int(window * 0.5) + (window % 2)
    uphalf = int(window*0.5)

    low_bound = np.max((target_idx[i] - lohalf, 0))
    hi_bound = np.min((target_idx[i] + uphalf, nener))

    if low_bound == 0:
        hi_bound = window
    if hi_bound == nener:
        low_bound = -window

    return low_bound, hi_bound


def _get_spectra_misc(data):
    """
    Calculate misc quantities on the data:

    1) Global mean energy across spectra for different
        disorder dealizations

    2) Variance of energy (Gamma**2) across flattened spectra
        for different disorder realizations

    3) Mean of variances (Gamma**2) for different disorder
        for different disorder realizations

    4) and 5): Average minimum and maximum energy

    6) Average median energy

    """
    mean_ene_global = np.mean(data)
    # standard deviation of the data (across all the samples)
    var_ene_global = np.std(data)**2
    # mean of the sample gammas
    var_ene_local = np.mean(np.std(data, axis=1)**2)
    # min energy
    min_ene = np.mean(np.min(data, axis=1))
    max_ene = np.mean(np.max(data, axis=1))

    median_ene = np.mean(np.median(data, axis=1))

    return (mean_ene_global, var_ene_global, var_ene_local,
            min_ene, max_ene, median_ene)


def _set_savepath(loadpath, os_sep = '/'):

    path = os.path.abspath(loadpath)
    split_ = path.split('/')[:-1] # drop the last bit of folder structure
    # print(split_)
    for i, val in enumerate(split_[::-1]):

        if ((i == 3) and (val == 'results')):

            split_[-(i+1)]= 'quick_txtfiles'

    path = os.path.join('/', *split_)
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


if __name__ == '__main__':

    rDict, rextra = arg_parser_general(_r_parse_dict)
    argsDict, extra = arg_parser([], [])

    r_step, = [
        rDict[key] for key in _r_parse_dict.keys()]

    print(r_step)

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    header = header.format(savepath, syspar, modpar )

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    # NOTE: this module does not create a hdf5 dataset containing
    # the results; instead, the results of sweeps for different
    # temperatures/energy densities are stored in an external file
    # try:
    file = glob.glob(f"{loadpath}/*.hdf5")[0]

    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        data = f['Eigenvalues'][:]

        attrs = dict(f['Eigenvalues'].attrs)

    # select the data from the energy window on which to calculate r
    Tlist = np.concatenate(([0.001, 0.01, 0.1], np.arange(0.2, 8.1, 0.1), [np.inf]))
    window = 500
    nener = data.shape[1]
    nsamples = data.shape[0]

    # ---------------------------------------------------
    #
    #  MISC QUANTITIES ON THE WHOLE SPECTRUM
    #
    # ---------------------------------------------------
    # calculate misc quantities on the data:
    #
    # 1) Global mean energy across spectra for different
    #    disorder dealizations
    #
    # 2) Variance of energy (Gamma**2) across flattened spectra
    #    for different disorder realizations
    #
    # 3) Mean of variances (Gamma**2) for different disorder
    #    for different disorder realizations
    #
    # 4) and 5): Average minimum and maximum energy
    #
    # 6) Average median energy
    #

    misc_vals = (mean_ene_global, var_ene_global,
                    var_ene_local, min_ene, max_ene, median_ene) = _get_spectra_misc(data)

    print('Misc spectral quantities:')
    print(
        f'E_mean: {mean_ene_global}, var_glob: {var_ene_global}, var_local: {var_ene_local}')
    print(
        f'min_ene: {min_ene}, max_ene: {max_ene}, median_ene: {median_ene}')

    # ------------------------------------------------
    #
    # START THE PROCEDURE
    #
    # ------------------------------------------------
    # a safety check
    if window > nener:
        window = nener

    

    results = []

    # TODO: make this parsable
    for temperature in Tlist:
        data_ = np.empty((nsamples, window))
        # calculate the target energy and the corresponding energy index
        # in the array of the eigenvalues
        print(f'T: {temperature}')
        target_ene, target_idx = _set_target_ene(temperature, data)
        mean_target_ene = np.mean(target_ene)
        for i, spectrum in enumerate(data):

            low_bound, hi_bound = _set_window(window, nener, target_idx, i)

            data_[i] = data[i][low_bound:hi_bound]

        mean_ener = np.mean(data_)
        spc = Spectra(data_)
        gap_mean, gap_dev = spc.gap_avg()
        print((f'T: {temperature}, tar_ene: {mean_target_ene}, \n'
                f'window_ene: {mean_ener}, r_mean: {gap_mean}, r_std: {gap_dev}'))

        results.append([temperature, mean_target_ene,
                        mean_ener, gap_mean, gap_dev, window, nener, nsamples,
                        *misc_vals])
    
    results = np.array(results)
    path_ = _set_savepath(loadpath)

    head, savename = os.path.split(file)

    savename = savename.replace('eigvals', f'r_microcan_window_{window}')
    savename = savename.replace('.hdf5', '.txt')
    savename = f'{path_}/{savename}'

    print(savename)
    np.savetxt(savename, results, header=header, footer=footer)

    # except IndexError:
    #     pass
