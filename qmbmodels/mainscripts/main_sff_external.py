#!/usr/bin/env python

"""
This module provides utilities for calculating sff data
and then storing them together with their corresponding
metadata in the hdf5 format. The code below relies
heavily on the tools from the spectral_stats package,
which can be obtained from:
https://github.com/JanSuntajs/spectral_statistics_tools


"""

import os
import numpy as np
import glob
import h5py

from spectral_stats.spectra import Spectra

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general

_sff_keys = ['sff_min_tau', 'sff_max_tau', 'sff_n_tau',
             'sff_eta', 'sff_unfolding_n', 'sff_filter']

_sff_parse_dict = {'sff_min_tau': [float, -5],
                   'sff_max_tau': [float, 2],
                   'sff_n_tau': [int, 1000],
                   'sff_eta': [float, 0.5],
                   'sff_unfolding_n': [int, 3],
                   'sff_filter': [str, 'gaussian']}

# which attributes of the Spectra class instance to exclude
# from the final hdf5 file

_filt_exclude = []  # ['filter', 'dims']
# which attributes to consider as separate datasets
_misc_include = []  # ['mean_ener', 'sq_ham_tr', 'ham_tr_sq', 'gamma', ]

# sfflist text descriptor
sfflist_desc = """
This string provides a textual descriptor of the
'SFF_spectra' hdf5 dataset. sfflist is a ndarray
of the shape (nsamples + 1, len(taulist)) where
nsamples is the number of different disorder
realizations for which the energy spectra of the
quantum hamiltonians have been calculated, and
len(taulist) is the number of tau values for which
the spectral form factor has been evaluated.
The first (zeroth, in python's numbering) entry
of the sfflist array is the list of tau values.
Other entries are calculated according to the
formula:


sfflist[m + 1, n] = np.sum(weights * np.exp(-1j * taulist[n] * spectrum[m]))

Where spectrum[m] is the m-th entry in the array of the
energy spectra and taulist[n] is the n-th entry in the
array of tau values. 'weights' is an array of some
multiplicative prefactors. No additional operations
have been performed on the spectra so that one can
also calculate averages, standard deviations and other
quantities of interest, if so desired. 'np' prefix
stands for the numpy python library, which we used
in the calculation.

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of the Spectral form
factor.

"""

sff_desc = """
This string provides a textual description of the
'SFF_spectrum' hdf5 dataset. sff is a ndarray of the
shape (3, len(taulist)) where len(taulist) stands for
the number of tau values at which sff has been
evaluated. Entries in the sff array:

sff[0]: taulist -> tau values, at which sff was evaluated
sff[1]: sff with the included disconnected part. This
        quantity is calculated according to the definition:
        np.mean(np.abs(sfflist)**2, axis=0). Here, sfflist
        is an array of sff spectra obtained for different
        disorder realizations.
sff[2]: disconnected part of the sff dependence, obtained
        according to the definition:
        np.abs(np.mean(sfflist, axis=0))**2

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of the Spectral form
factor.
"""


# spectral_widths
spec_widths = [
    [0., 1],
    [0.45, 0.55],
    [0.495, 0.505],
    [0.4995, 0.5005]
]

if __name__ == '__main__':

    sffDict, sffextra = arg_parser_general(_sff_parse_dict)
    argsDict, extra = arg_parser([], [])

    min_tau, max_tau, n_tau, eta, unfold_n, sff_filter = [
        sffDict[key] for key in _sff_parse_dict.keys()]

    sff_filter = sff_filter.lower()
    print(min_tau, max_tau, n_tau, sff_filter)

    savepath = argsDict['results']
    print(savepath)

    sff_folder = f'{savepath}/sff_ext'
    if not os.path.isdir(sff_folder):

        os.makedirs(sff_folder)
    try:
        file = glob.glob(f"{savepath}/*.hdf5")[0]
        print(('Starting sff analysis in the {}'
               'folder!').format(savepath))

        with h5py.File(file, 'r', libver='latest', swmr=True) as f:

            data = f['Eigenvalues'][:]

            attrs = dict(f['Eigenvalues'].attrs)
            attrs.update(sffDict)
            # create the spectra class instance
            taulist = np.logspace(min_tau, max_tau, n_tau)

            # if the sff spectrum dataset does not yet exist, create it
            # perform the calculations
            for spc_width in spc_widths:
                spc = Spectra(data)
                # if sff_filter == 'gaussian':
                spc.spectral_width = tuple(spectral_width)

                spc.spectral_unfolding(
                    n=unfold_n, merge=False, correct_slope=True)
                spc.get_ham_misc(individual=True)
                spc.spectral_filtering(filter_key=sff_filter, eta=eta)

                # calculate the SFF
                spc.calc_sff(taulist, return_sfflist=False)
                # gather the results
                sffvals = np.array([spc.taulist, spc.sff, spc.sff_uncon])

                D_eff = spc.filt_dict['dims_eff']
                normal_uncon = spc.filt_dict['normal_uncon']
                normal_con = spc.filt_dict['normal_con']

                sff_discon = spc.sff / D_eff
                sff_con = (spc.sff - (normal_con / normal_uncon) *
                           spc.sff_uncon) / D_eff

                sffvals_rescaled = np.array(
                    [spc.taulist / (2 * np.pi), sff_discon, sff_con])
                # prepare additional attributes
                filt_dict = {key: spc.filt_dict[key] for key in spc.filt_dict
                             if key not in _filt_exclude}
                misc_dict = {key: spc.misc_dict[key]
                             for key in spc.misc_dict if
                             key not in _misc_include}
                misc0 = spc.misc0_dict.copy()
                misc0_keys = [key for key in misc0]

                for key in misc0_keys:
                    misc0[key + '0'] = misc0.pop(key)
                attrs.update(spc.unfold_dict.copy())

                for dict_ in (misc_dict, filt_dict, misc0):
                    attrs.update(dict_)

                attrs.update({'nener': spc.nener, 'nsamples': spc.nsamples,
                              'nener0': spc._nener,
                              'nsamples0': spc._nsamples})

                name_spectrum = ('SFF_spectrum_perc_{:.5f}_eta'
                                 '_{:.4f}_filter_{}_{}_{}.npz').format(
                    spc_width[1] - spc_width[0],
                    eta, sff_filter, argsDict['syspar'], argsDict['modpar'])

                filedict = {'sff_raw': sffvals,
                            'sff_rescaled': sffvals_rescaled}
                np.savez(f'{sff_folder}/{name_spectrum}', **attrs, **filedict)
    except IndexError:
        print('No hdf5 file in the {} folder! Exiting.'.format(savepath))
        pass
