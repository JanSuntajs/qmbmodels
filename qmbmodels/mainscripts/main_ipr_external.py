#!/usr/bin/env python

"""
This module provides utilities for calculating
the mean mean ipr and the mean entanglement
entropy across different disorder
realizations. The data are saved into external
folder as a .txt file.
This implementation uses tools from the
spectral_statistics_tools package
available at:

https://github.com/JanSuntajs/spectral_statistics_tools

We calculate the ipr and entanglement entropy for a range
of q values:

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
     2., 2.4, 2.6, 2.8., 3., 4., 5., 6.];

Evidently, q=1 is missing above, since we need to use different
definitions both for IPR and the entanglement entropy. For the
former, we calculate the corresponding participation entropy
as the Shannon entropy of the eigenstate probability distribution.
Note that calculating the IPR for q=1 makes no sense as it would
trivially yield normalization. For the entanglement entropy,
we calculate the standard von Neumann entanglement entropy, which
is the limiting case of the Renyi entanglement entropy as q goes
to 1.


# ---------------------
#
# CALLING THE SCRIPT
#
# ---------------------

Command-line argument for calling this script following
the corresponding submission script:

ipr_external

Hence calling the program within qmbmodels would be:

python <submission_script.py> ipr_external

or, to perform the main ipr calculation followed by
this post-processing step:

python <submission_script.py> diag_ipr ipr_external

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
expansion of \ksi_\alpha in the computational basis.
If the file has only IPR_average* in the name, then
the IPR as defined above is being averaged. In case
the file name is IPR_log_average*, then \log(ipr)
is being averaged. This also holds for the q=1 limit.


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

1-27) Eigenstate IPR values averaged over different disorder realizations
   in the same manner as energies. In column order, the following
   q-values (see the header) are saved:
   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
   1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.,
        2.2,      2.4,      2.6,      2.8,      3.0
        4.,        5.,      6.,
   In case the file name is IPR_log_average*, then the logarithms
   of ipr are averaged over disorder realizations.

28) Shannon entropy as the limit in the q -> 1 case.
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

1-27) Von Renyi eigenstate entanglement entropies (1/ (1-q))\log\sum \lambda_p^q
   in the same manner as energies. In column order, the following
   q-values (see the header) are saved:
   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
   1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.,
        2.2,      2.4,      2.6,      2.8,      3.0
        4.,       5.,       6.
28) Von Neumann entanglement entropy as -\sum \lambda * \log\lambda 

29) Schmidt gaps -> differences between the largest eigenvalues of the reduced
    density matrix averaged over disorder and eigenstates.


"""

header_mean = """
This file provides data on the averaged Renyi_2 entropy
defined as - \log \sum \lambda^2_p

where \lambda^2_p are the eigenvalues of the reduced
density matrix for a homogenous bipartition of
p farthermost spins and the remainder of the system.
One can infer the value of p from the filename.

In this file, we focus on the statistical analysis of the
spectra; for energies, Renyi_2 entropies and Schmidt gaps,
we (in the said order) provide the following data:

- sample means (e. g., means for each sample)
- sample standard deviations (e. g., standard deviations of the
data across a sample for different samples)
- global means (e. g., means across both states and disorder realizations)
- global standard deviations (e. g., standard deviation across both
states and disorder realizations.)


Location of the initial .hdf5 datafile: {} \n
System parameters: {} \n
Module parameters: {} \n

"""

footer_mean = """
In this footer, we provide a description on the contents of the
columns in the file.

Energy data
0) - 3): mean energy for each sample, standard deviation for each sample,
global mean and global standard deviation. The last two are rescaled such
that their dimension matches the first two columns

Renyi_2 entropy data (S_2)
4) - 7): mean S_2 for each sample, standard deviation of S_2 for each sample
global mean and global standard deviation of S_2

Schmidt gap (SG) data:
8) - 11): mean SG for each sample, standard deviation of SG for each sample,
global mean and global standard deviation of SG.

"""


headers = [header, header_eentro, header_mean]

qlist = np.append(np.arange(0.1, 2.1, 0.1), np.arange(2.2, 3.2, 0.2))
qlist = np.append(qlist, np.arange(4, 7, 1))
qlist = np.delete(qlist, 9)

# which values to consider in the mean analysis
mean_qlist = [0.1, 0.5, 1, 2]
# qlist_eentro = np.append(np.arange(0.1, 1., 0.1), 2)

plist = [1, 2, 3, 4]

# entanglement entropy analysis
eentro_string = 'EENTRO_RENYI_p_{:d}_q_{:.2f}'

eentro_vn_string = 'EENTRO_VN_p_{:d}'

schmidt_gap_string = 'SCHMIDT_GAP_p_{:d}'


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


def _get_mean_vals(data):
    """
    An internal helper routine
    for calculating (in that order):
        the sample mean (mean of the
        sample)
        the sample standard deviation
        the global mean
        the global standard deviation

    Parameters:

    data: ndarray, 2D

    """

    sample_mean = np.mean(data, axis=1)
    sample_std = np.std(data, axis=1)

    global_mean = np.ones_like(
        sample_mean)*np.mean(data,)
    global_std = np.ones_like(
        sample_mean)*np.std(data,)

    return sample_mean, sample_std, global_mean, global_std


if __name__ == '__main__':

    # rDict, rextra = arg_parser_general(_r_parse_dict)
    argsDict, extra = arg_parser([], [])

    # r_step, = [
    #     rDict[key] for key in _r_parse_dict.keys()]

    # print(r_step)

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

    mean_dict = {p: {} for p in plist}

    with h5py.File(file, 'a', libver='latest', swmr=True) as f:

        # -------------------------------
        #
        #   IPR data
        #
        # -------------------------------
        try:

            data = f['EIG_IPR'][:]

            attrs = dict(f['Eigenvalues'].attrs)

            for q_ in np.append(qlist, 1):

                if q_ != 1:
                    iprlist.append(f[f'IPR_q_{q_:.2f}'][:])
                else:
                    iprlist.append(f['ENTRO_PART_q_1.00'][:])

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

                # schmidt gap data
                dset_sg_ = f[schmidt_gap_string.format(p)][:]
                # von neumann entropy
                dset_vn_ = f[eentro_vn_string.format(p)][:]

                # --------------------------------------
                #
                # energies and schmidt gaps are
                # independent of q so no need for
                # calculations in a loop
                #
                # --------------------------------------

                # --------------------------------------
                # energies
                # --------------------------------------

                (sample_mean_ene, sample_std_ene,
                 global_mean_ene, global_std_ene) = _get_mean_vals(data)

                # ----------------------------------------
                # schmidt gap
                # ----------------------------------------
                (sample_mean_sg, sample_std_sg,
                 global_mean_sg, global_std_sg) = _get_mean_vals(dset_sg_)

                for q in np.append(qlist, 1):

                    if q != 1:
                        dset_ = f[eentro_string.format(p, q)][:]
                        eentro_dict[p].append(dset_)

                    else:
                        dset_ = dset_vn_

                    if q in mean_qlist:

                        mean_dict[p][q] = []

                        # ----------------------------------------
                        # renyi_2 entropies stats analysis
                        # ----------------------------------------

                        # mean and std for each individual sample
                        (sample_mean, sample_std,
                         global_mean, global_std) = _get_mean_vals(dset_)

                        mean_dict[p][q] += [sample_mean_ene, sample_std_ene,
                                            global_mean_ene, global_std_ene,
                                            sample_mean, sample_std,
                                            global_mean,
                                            global_std, sample_mean_sg,
                                            sample_std_sg, global_mean_sg,
                                            global_std_sg]

                eentro_dict[p].append(dset_vn_)
                eentro_dict[p].append(dset_sg_)

        except KeyError:
            print('EENTRO related keys not present!')

    nener = data.shape[1]
    nsamples = data.shape[0]

    # ------------------------------------------------------
    #
    # MEAN VALUES
    #
    # ------------------------------------------------------

    # take the mean of the relevant quantities to be saved
    mean_ener = np.mean(data, axis=0)
    mean_ipr = [np.mean(ipr_, axis=0) for ipr_ in iprlist]
    # mean_log_ipr = [np.mean(np.log(ipr_), axis=0) if 'ENTRO_PART' not in
    #                 ipr_ else np.mean(ipr_, axis=0) for ipr_ in iprlist]
    mean_log_ipr = [np.mean(np.log(ipr_), axis=0) for ipr_ in iprlist[:-1]]
    # special case (the limiting one)
    mean_log_ipr.append(np.mean(iprlist[-1], axis=0))

    mean_eentro = {}
    for p in plist:

        mean_eentro[p] = [np.mean(eentro,
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
    # ----------------------------------
    # save ipr results
    # ----------------------------------
    resultlist = [results, results_log]
    savenames = ['IPR_average', 'IPR_log_average']

    for i, result in enumerate(resultlist):

        savename_ = savename.replace('eigvals', savenames[i])
        savename_ = savename_.replace('.hdf5', '.txt')
        savename_ = f'{path_}/{savename_}'
        print(savename_)
        np.savetxt(savename_, result, header=headers[0], footer=footer)

    # -----------------------------------
    # save general entropy results
    # -----------------------------------
    for p in plist:

        results = np.vstack((mean_ener, mean_eentro[p])).T

        savename_ = savename.replace('eigvals', f'EENTRO_p_{p}_average_')
        savename_ = savename_.replace('.hdf5', '.txt')
        savename_ = f'{path_}/{savename_}'
        np.savetxt(savename_, results, header=headers[1],
                   footer=footer_eentro)

    # --------------------------------------
    # save entropy statistical results
    # --------------------------------------
    for p in plist:

        for q in mean_qlist:
            results = np.array(mean_dict[p][q]).T

            savename_ = savename.replace('eigvals', f'EENTRO_{q}_STATS_p_{p}_')
            savename_ = savename_.replace('.hdf5', '.txt')
            savename_ = f'{path_}/{savename_}'
            np.savetxt(savename_, results, header=headers[2],
                       footer=footer_mean)

    # except IndexError:
    #     pass
