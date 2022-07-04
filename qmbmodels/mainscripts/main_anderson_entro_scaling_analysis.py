#!/usr/bin/env python

"""
This module provides utilities for calculating
the mean entanglement entropies of the Anderson model
for different scaling fractions

f = V_A / V where V_A is the subsystem volume and
V is the total volume of the system. Currently, only
the homogenous bipartitions are allowed and supported.

"""

import os
import numpy as np
import glob
import h5py

from scipy.stats import norm
from spectral_stats.spectra import Spectra


from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general
from qmbmodels.mainscripts.main_ipr_external import _set_savepath


# -------------------------------------------------
#
#
#  headers/footers
#
#
# -------------------------------------------------

header = """
This file provides data on averaged many-body
eigenstate von Neumann entanglement entropy
for the quadratic (Anderson-like) models and
its scaling with the subsystem fraction.


Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n
"""

footer = """
In this footer, we provide a description on the contents of the
columns in the file.

0) System fractions f = V_A / V where V_A is the subsystem volume
   and V the system volume

1) Mean S (entanglement entropy) for the corresponding f value.

2) Mean S scaled by (0.5 V * np.log(2)) for easier comparison
   with the maximally entangled results

3) Standard deviation of S (not the scaled variant)

4) Number of energy states participating in the calculation of the
   mean value.

5) Filling fraction (1/2 at half-filling -> the number of particles)

6) V, the volume of the full system (also the Hilbert space dimension.)

7) Number of random samples used in the disorder averaging.


"""



_eentro_parse_dict = {'eentro_nstates': [int, -1],
                      'eentro_filling': [float, 0.5],
                      'eentro_partition': [float, 0.5],
                      'eentro_grandcanonical': [int, 1]}

# dataset name
entanglement_name = ('Entropy_noninteracting_'
                     'nstates_{}_'
                     'partition_size_{:.2f}_'
                     'filling_{:.2f}')

if __name__ == '__main__':

    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']
    
    header = header.format(savepath, syspar, modpar)
    eentroDict, eentro_extra = arg_parser_general(_eentro_parse_dict)

    # number of many-body states for which we calculate
    # the entropy
    eentro_nstates = eentroDict['eentro_nstates']
    # filling fraction - ratio of occupied states
    # compared to the volume of the system
    filling = eentroDict['eentro_filling']

    # partition_fraction = eentroDict['eentro_partition']

    gc = eentroDict['eentro_grandcanonical']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    # NOTE: this module does not create a hdf5 dataset containing
    # the results; instead, the results of sweeps for different
    # temperatures/energy densities are stored in an external file
    # try:
    file = glob.glob(f"{loadpath}/*.hdf5")[0]

    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        # -------------------------------
        #
        #   IPR data
        #
        # -------------------------------

        eentro_mean = []
        eentro_std = []

        frac_arr = np.arange(0.1, 0.55, 0.05)
        for vol_fraction in frac_arr:

            dset_name = entanglement_name.format(
                eentro_nstates, vol_fraction, filling)

            nstates = f[dset_name].attrs['nener']
            print(nstates)
            try:

                data = f[dset_name][:]

                
                nsamples = data.shape[0]
                nentro = data.shape[1]

                eentro_mean_ = np.mean(data)
                eentro_mean.append(eentro_mean_)
                eentro_std.append(np.std(data))

            except KeyError:
                print('Entanglement entropy key not present!')


        frac_arr = np.append(frac_arr, 1 - frac_arr[:-1][::-1])
        eentro_mean = np.append(eentro_mean, eentro_mean[:-1][::-1])
        eentro_std = np.append(eentro_std, eentro_std[:-1][::-1])
        eentro_mean_scaled = eentro_mean / (np.log(2) * nstates * 0.5)
        
        eentro_nstates_ = np.ones_like(frac_arr) * eentro_nstates
        ful_vol_ = np.ones_like(frac_arr) * nstates
        nsamples_ = np.ones_like(frac_arr) * nsamples
        filling_ = np.ones_like(frac_arr) * filling

    # ------------------------------------------------
    #
    # START THE PROCEDURE
    #
    # ------------------------------------------------

    # save the results for ipr and log_ipr (of the individual
    # spectra)

    results = np.vstack((frac_arr, eentro_mean, eentro_mean_scaled, eentro_std,
                         eentro_nstates_, filling_, ful_vol_, nsamples_)).T

    path_ = _set_savepath(loadpath)

    head, savename = os.path.split(file)
    # ----------------------------------
    # save ipr results
    # ----------------------------------
    
    savename_ = 'eentro_f_scaling'

    # for i, result in enumerate(resultlist):

    savename_ = savename.replace('eigvals', savename_)
    savename_ = savename_.replace('.hdf5', '.txt')
    savename_ = f'{path_}/{savename_}'
    print(savename_)
    np.savetxt(savename_, results, header=header, footer=footer)

