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
for the interacting (many-body) models and
its scaling with the subsystem fraction.


Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n
"""

footer = """
In this footer, we provide a description on the contents of the
columns in the file.

0) Subsystem f = L_A / L

1) Mean S (entanglement entropy) for the corresponding f value.

2) Mean S scaled by (0.5 L * np.log(2)) for easier comparison
   with the maximally entangled results

3) Standard deviation of S (not the scaled variant)

4) Number of energy states participating in the calculation of the
   mean value.

5) Subsystem volume L_A

6) L, the volume of the full system (also the Hilbert space dimension.)

7) Number of random samples used in the disorder averaging.


"""


def _get_subs_size(attr_name):

    size = int(attr_name.strip('_partial').strip('Entropy_scaling_'))

    return size


_eentro_parse_dict = {}

# dataset name
entro_name = 'Entropy_scaling'

if __name__ == '__main__':

    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    header = header.format(savepath, syspar, modpar)
    eentroDict, eentro_extra = arg_parser_general(_eentro_parse_dict)

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    # NOTE: this module does not create a hdf5 dataset containing
    # the results; instead, the results of sweeps for different
    # temperatures/energy densities are stored in an external file
    # try:
    file = glob.glob(f"{loadpath}/*.hdf5")[0]

    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        # get subsystem sizes and corresponding keys
        entro_keys = [key for key in f.keys() if entro_name in key]
        entro_keys = sorted(entro_keys, key=_get_subs_size)
        subsizes = np.array([_get_subs_size(key) for key in entro_keys])

        eentro_mean = []
        eentro_std = []

        for i, dset_name in enumerate(entro_keys):
            # print(dset_name)
            try:

                data = f[dset_name][:]
                vol_ = f[dset_name].attrs['L']

                nsamples = data.shape[0]
                nener = data.shape[1]

                eentro_mean_ = np.mean(data)
                eentro_mean.append(eentro_mean_)
                eentro_std.append(np.std(data))

            except KeyError:
                print('Entanglement entropy key not present!')

        eentro_mean_scaled = eentro_mean / (np.log(2) * vol_ * 0.5)

        ful_vol_ = np.ones_like(subsizes) * vol_
        nsamples_ = np.ones_like(subsizes) * nsamples
        nener_ = np.ones_like(subsizes) * nener
    # ------------------------------------------------
    #
    # START THE PROCEDURE
    #
    # ------------------------------------------------

    # save the results for ipr and log_ipr (of the individual
    # spectra)

    results = np.vstack(
        (subsizes/vol_, eentro_mean, eentro_mean_scaled,
        eentro_std, nener_, subsizes, ful_vol_, nsamples_)).T

    path_ = _set_savepath(loadpath)

    head, savename = os.path.split(file)
    # ----------------------------------
    # save ipr results
    # ----------------------------------

    savename_ = 'eentro_interacting_f_scaling'

    # for i, result in enumerate(resultlist):


    savename_ = savename.replace('eigvals', savename_)
    savename_ = savename_.replace('.hdf5', '.txt')
    savename_ = f'{path_}/{savename_}'
    print(savename_)
    np.savetxt(savename_, results, header=header, footer=footer)
