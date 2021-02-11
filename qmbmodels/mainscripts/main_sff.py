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
             'sff_unfolding_n', 'sff_filter', 'sff_eta_min',
             'sff_eta_max', 'sff_eta_step']

_sff_parse_dict = {
    'sff_min_tau': [float, -5],  # minimum tau exponent
    'sff_max_tau': [float, 2],  # upper tau exponent (log powers)
    'sff_n_tau': [int, 1000],  # number of tau values
    # the degree of the unfolding poly.
    'sff_unfolding_n': [int, 3],
    # what kind of filtering to perform
    'sff_filter': [str, 'gaussian'],
    'sff_eta_min': [float, 0.5],  # eta minimum value
    'sff_eta_max': [float, 0.5],  # eta maximum value
    # incremental step for increasing eta val.
    'sff_eta_step': [float, 0.0]
}

# which attributes of the Spectra class instance to exclude
# from the final hdf5 file

_filt_exclude = ['filter', 'dims']
# which attributes to consider as separate datasets
_misc_include = ['mean_ener', 'sq_ham_tr', 'ham_tr_sq', 'gamma',
                 ]

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

moments_desc = """
This string provides a textual description of the
'SFF_moments' hdf5 dataset. SFF_moments is a ndarray of the
shape (11, len(taulist)) where len(taulist) stands for
the number of tau values at which sff has been
evaluated. Entries in the sff array:

moments[0]: taulist -> tau values, at which sff was evaluated

moments[1:]: moments of the sff calculated according to:

            I_m = np.mean(np.abs(sfflist)**(2*m), axis=0) /
                  np.mean(np.abs(sfflist)**2, axis=0)**m
            where m takes interval values in the range
            from 1 to 10.

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of the Spectral form
factor.


"""

if __name__ == '__main__':

    sffDict, sffextra = arg_parser_general(_sff_parse_dict)
    argsDict, extra = arg_parser([], [])

    min_tau, max_tau, n_tau, unfold_n, sff_filter, \
        min_eta, max_eta, step_eta = [
            sffDict[key] for key in _sff_parse_dict.keys()]

    if max_eta < min_eta:
        raise ValueError('min_eta larger than max eta!')
    if step_eta < 0.:
        raise ValueError('step_eta must be nonnegative!')
    # number of eta points
    n_eta = np.int((max_eta - min_eta) / step_eta) + 1
    etavals = np.linspace(min_eta, max_eta, n_eta)

    sff_filter = sff_filter.lower()
    print(min_tau, max_tau, n_tau, sff_filter)

    savepath = argsDict['results']
    print(savepath)

    try:
        file = glob.glob(f"{savepath}/*.hdf5")[0]
        print(('Starting sff analysis in the {}'
               'folder!').format(savepath))

        for eta in etavals:
            with h5py.File(file, 'a', libver='latest') as f:
                print(('Starting the sff calculation process... '
                       'Loading the eigenvalues.'))
                data = f['Eigenvalues'][:]

                attrs = dict(f['Eigenvalues'].attrs)
                attrs.update(sffDict)
                # create the spectra class instance
                taulist = np.logspace(min_tau, max_tau, n_tau)

                # if the sff spectrum dataset does not yet exist, create it
                # perform the calculations
                spc = Spectra(data)
                if sff_filter == 'gaussian':
                    spc.spectral_width = (0., 1.)
                elif sff_filter == 'identity':
                    spc.spectral_width = (0.5 * eta, 1 - 0.5 * eta)
                spc.spectral_unfolding(
                    n=unfold_n, merge=False, correct_slope=True)
                spc.get_ham_misc(individual=True)
                print(('Performing spectral filtering... '
                       'Filter used: {}. eta: {}').format(sff_filter, eta))
                spc.spectral_filtering(filter_key=sff_filter, eta=eta)
                sfflist = np.zeros(
                    (spc.nsamples + 1, len(taulist)), dtype=np.complex128)

                momentlist = np.zeros((11, len(taulist)), dtype=np.float64)
                sfflist[0] = taulist
                # calculate the SFF
                sfflist[1:, :] = spc.calc_sff(taulist, return_sfflist=True)
                # gather the results
                sffvals = np.array([spc.taulist, spc.sff, spc.sff_uncon])

                # also calculate the first 10 moments for the distribution:
                for m in range(1, 11, 1):

                    denumerator = np.mean(
                        np.abs(sfflist[1:, :])**(2 * m), axis=0)
                    denominator = np.mean(np.abs(sfflist[1:, :]**2), axis=0)**m

                    moment = denumerator / denominator
                    momentlist[m, :] = moment

                print('Finished with calculations... Storing results!')
                # prepare additional attributes
                filt_dict = {key: spc.filt_dict[key] for key in spc.filt_dict
                             if key not in _filt_exclude}
                misc_dict = {key: spc.misc_dict[key]
                             for key in spc.misc_dict if key
                             not in _misc_include}
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

                eta_filt_desc = '_eta_{:.4f}_filter_{}'.format(eta,
                                                               sff_filter)

                # add the actual sff values
                key_spectra = 'SFF_spectra{}'.format(eta_filt_desc)
                key_spectrum = 'SFF_spectrum{}'.format(eta_filt_desc)
                key_moments = 'SFF_moments{}'.format(eta_filt_desc)

                if key_spectra not in f.keys():
                    print('Creating datasets {} and {}'.format(
                        key_spectra, key_spectrum, key_moments))

                    f.create_dataset(key_spectra, data=sfflist,
                                     maxshape=(None, None))
                    f.create_dataset(key_spectrum, data=sffvals,
                                     maxshape=(3, None))

                    f[key_spectra].attrs['Description'] = sfflist_desc
                    f[key_spectrum].attrs['Description'] = sff_desc

                else:

                    print('Updating datasets {} and {}'.format(
                        key_spectra, key_spectrum))
                    f[key_spectra].resize(sfflist.shape)
                    f[key_spectrum].resize(sffvals.shape)
                    f[key_moments].resize(momentlist.shape)

                    f[key_spectra][()] = sfflist
                    f[key_spectrum][()] = sffvals
                    f[key_moments][()] = momentlist

                if key_moments not in f.keys():
                    f.create_dataset(key_moments, data=momentlist,
                                     maxshape=(11, None))
                    f[key_moments].attrs['Description'] = moments_desc

                else:
                    print('Updating datasets {}'.format(
                        key_moments))

                    f[key_moments].resize(momentlist.shape)
                    f[key_moments][()] = momentlist

                _misc_include_ = []
                # data which led to the SFF calculation
                for key in _misc_include:
                    _misc_include_.append(key + eta_filt_desc)
                    key_ = key + eta_filt_desc
                    if key_ not in f.keys():

                        f.create_dataset(
                            key_,
                            data=spc.misc_dict[key], maxshape=(None,))
                    else:
                        f[key_].resize(spc.misc_dict[key].shape)
                        f[key_][()] = spc.misc_dict[key]

                # append the attributes
                for key1 in [key_spectra, key_spectrum,
                             key_moments] + _misc_include_:
                    for key2, value in attrs.items():
                        f[key1].attrs[key2] = value

                print('Finished!')
    except IndexError:
        print('No hdf5 file in the {} folder! Exiting.'.format(savepath))
        pass
