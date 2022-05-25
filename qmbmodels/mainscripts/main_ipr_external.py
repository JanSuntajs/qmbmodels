#!/usr/bin/env python

"""
This module provides utilities for calculating
the mean mean ipr across different disorder
realizations. The data are saved into external
folder as a .txt file.
This implementation uses tools from the
spectral_statistics_tools package
available at:

https://github.com/JanSuntajs/spectral_statistics_tools


"""

import os
import numpy as np
import glob
import h5py

from scipy.stats import norm
from spectral_stats.spectra import Spectra

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general

_r_keys = ['r_step']

_r_parse_dict = {'r_step': [float, 0.25], }

_r_name = 'r_data'
# sfflist text descriptor
header = """
This file provides data on averaged eigenstate
IPR (inverse participation ratio), defined
for an eigenstate \ksi_\alpha as

ipr_\alpha = \sum |c_i^(\alpha)|^4
where c_i^(\alpha) are the coefficients of the
expansio of \ksi_\alpha in the computational basis.


Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n
"""

footer = """
In this footer, we provide a description on the contents of the
columns in the file. 

0) Energies averaged across different disorder realizations
   (i-th) eigenlevel averaged across i-th eigenlevels of all
   realizations.

1) Eigenstate IPR values averaged over different disorder realizations
   in the same manner as energies.



"""




def _set_savepath(loadpath, os_sep='/'):

    path = os.path.abspath(loadpath)
    split_ = path.split('/')[:-1]  # drop the last bit of folder structure
    # print(split_)
    for i, val in enumerate(split_[::-1]):

        if ((i == 3) and (val == 'results')):

            split_[-(i+1)] = 'quick_txtfiles'

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

    header = header.format(savepath, syspar, modpar)

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    # NOTE: this module does not create a hdf5 dataset containing
    # the results; instead, the results of sweeps for different
    # temperatures/energy densities are stored in an external file
    # try:
    file = glob.glob(f"{loadpath}/*.hdf5")[0]

    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        data = f['Eigenvalues'][:]

        attrs = dict(f['Eigenvalues'].attrs)

        try:

            ipr = f['IPR'][:]

        except KeyError:
            print('IPR key not present!')
            ipr = np.empty_like(data)

    nener = data.shape[1]
    nsamples = data.shape[0]


    mean_ener = np.mean(data, axis=0)
    mean_ipr = np.mean(ipr, axis=0)

    # ------------------------------------------------
    #
    # START THE PROCEDURE
    #
    # ------------------------------------------------


    results = np.vstack((mean_ener, mean_ipr)).T
    path_ = _set_savepath(loadpath)

    head, savename = os.path.split(file)

    savename = savename.replace('eigvals', f'IPR_average_')
    savename = savename.replace('.hdf5', '.txt')
    savename = f'{path_}/{savename}'

    print(savename)
    np.savetxt(savename, results, header=header, footer=footer)

    # except IndexError:
    #     pass
