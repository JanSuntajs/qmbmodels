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

# -------------------------------------------------
#
#
# IPR headers/footers
#
#
# -------------------------------------------------

header = """
This file provides data on averaged eigenstate
IPR (inverse participation ratio), defined
for an eigenstate \ksi_\alpha as

ipr_\alpha = \sum |c_i^(\alpha)|^2q
where c_i^(\alpha) are the coefficients of the
expansio of \ksi_\alpha in the computational basis.
If the file has only IPR_average* in the name, then
the IPR as defined above is being averaged. In case
the file name is IPR_log_average*, then \log(ipr)
is being averaged.


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

1-10) Eigenstate IPR values averaged over different disorder realizations
   in the same manner as energies. In column order, the following
   q-values (see the header) are saved:
   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.
   In case the file name is IPR_log_average*, then the logarithms
   of ipr are averaged over disorder realizations.


"""

# -------------------------------------------------------------
#
#
# EENTRO headers/footers
#
#
# --------------------------------------------------------------

header_eentro = """
This file provides data on averaged generalized
eigenstate entanglement entropy Q_q^(p) which we
define as 
\sum \lambda^q_p
where \lambda^q_p are the eigenvalues of the reduced
density matrix for a homogenous bipartition of
p farthermost spins and the remainder of the system.
One can infer the value of p from the filename.


Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n
"""

footer_eentro = """
In this footer, we provide a description on the contents of the
columns in the file.

0) Energies averaged across different disorder realizations
   (i-th) eigenlevel averaged across i-th eigenlevels of all
   realizations.

1-11) Generalized eigenstate entanglement entropies \sum \lambda_p^q
   in the same manner as energies. In column order, the following
   q-values (see the header) are saved:
   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2.


"""

headers = [header, header_eentro]

qlist = np.append(np.arange(0.1, 1., 0.1), 2)
qlist_eentro = np.append(np.arange(0.1, 1.1, 0.1), 2)

plist = [1, 2, 3, 4]

eentro_string = 'EENTRO_Q_p_{:d}_q_{:.2f}'


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

    for i, header_ in enumerate(headers):
        headers[i] = header_.format(savepath, syspar, modpar)

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    # NOTE: this module does not create a hdf5 dataset containing
    # the results; instead, the results of sweeps for different
    # temperatures/energy densities are stored in an external file
    # try:
    file = glob.glob(f"{loadpath}/*.hdf5")[0]

    iprlist = []
    eentro_dict = {p: [] for p in plist}
    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        # -------------------------------
        #
        #   IPR data
        #
        # -------------------------------
        try:

            data = f['EIG_IPR'][:]

            attrs = dict(f['Eigenvalues'].attrs)

            for q_ in qlist:

                iprlist.append(f[f'IPR_q_{q_:.2f}'][:])

        except KeyError:
            print('IPR related keys not present!')

            data = np.empty_like()
            iprlist = []

        # ----------------------------------------
        #
        # EENTRO data
        #
        # ---------------------------------------

        try:

            attrs = dict(f['Eigenvalues'].attrs)

            for p in plist:

                for q in qlist_eentro:

                    eentro_dict[p].append(f[eentro_string.format(p, q)][:])

        except KeyError:
            print('EENTRO related keys not present!')

    nener = data.shape[1]
    nsamples = data.shape[0]

    mean_ener = np.mean(data, axis=0)
    mean_ipr = [np.mean(ipr_, axis=0) for ipr_ in iprlist]
    mean_log_ipr = [np.mean(np.log(ipr_), axis=0) for ipr_ in iprlist]

    mean_eentro = {}
    for p in plist:

        mean_eentro[p] = [np.mean(np.log(eentro),
                                  axis=0) for eentro in eentro_dict[p]]

    # ------------------------------------------------
    #
    # START THE PROCEDURE
    #
    # ------------------------------------------------

    # save the results for ipr and log_ipr (of the individual
    # spectra)

    results = np.vstack((mean_ener, mean_ipr)).T
    results_log = np.vstack((mean_ener, mean_log_ipr)).T
    path_ = _set_savepath(loadpath)

    head, savename = os.path.split(file)

    resultlist = [results, results_log]
    savenames = ['IPR_average', 'IPR_log_average']

    for i, result in enumerate(resultlist):

        savename_ = savename.replace('eigvals', savenames[i])
        savename_ = savename_.replace('.hdf5', '.txt')
        savename_ = f'{path_}/{savename_}'
        print(savename_)
        np.savetxt(savename_, result, header=headers[0], footer=footer)

    for p in plist:

        results = np.vstack((mean_ener, mean_eentro[p])).T

        savename_ = savename.replace('eigvals', f'EENTRO_p_{p}_average_')
        savename_ = savename_.replace('.hdf5', '.txt')
        savename_ = f'{path_}/{savename_}'
        np.savetxt(savename_, results, header=headers[1],
                   footer=footer_eentro)

    # except IndexError:
    #     pass
