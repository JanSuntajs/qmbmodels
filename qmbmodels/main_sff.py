#!/usr/bin/env python

import os
import numpy as np
import glob
import h5py

from spectral_stats.spectra import Spectra

from utils import set_mkl_lib
from utils.cmd_parser_tools import arg_parser, arg_parser_general

_sff_keys = ['sff_min_tau', 'sff_max_tau', 'sff_n_tau',
             'sff_eta', 'sff_unfolding_n', 'sff_filter']

_sff_parse_dict = {'sff_min_tau': [float, -5],
                   'sff_max_tau': [float, 2],
                   'sff_n_tau': [int, 1000],
                   'sff_eta': [float, 0.5],
                   'sff_unfolding_n': [int, 3],
                   'sff_filter': [str, 'gaussian']}

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

if __name__ == '__main__':

    sffDict, sffextra = arg_parser_general(_sff_parse_dict)
    argsDict, extra = arg_parser([], [])

    min_tau, max_tau, n_tau, eta, unfold_n, sff_filter = [
        sffDict[key] for key in _sff_parse_dict.keys()]

    print(min_tau, max_tau, n_tau, sff_filter)

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a') as f:

            data = f['Eigenvalues'][:]

            attrs = dict(f['Eigenvalues'].attrs)
            attrs.update(sffDict)
            # create the spectra class instance
            taulist = np.logspace(min_tau, max_tau, n_tau)

            # if the sff spectrum dataset does not yet exist, create it

            spc = Spectra(data)
            spc.spectral_width = (0., 1.)
            spc.spectral_unfolding(n=unfold_n, merge=False, correct_slope=True)
            spc.get_ham_misc(individual=True)
            spc.spectral_filtering(filter_key=sff_filter, eta=eta)
            sfflist = np.zeros(
                (spc.nsamples + 1, len(taulist)), dtype=np.complex128)
            sfflist[0] = taulist
            sfflist[1:, :] = spc.calc_sff(taulist, return_sfflist=True)

            sffvals = np.array([spc.taulist, spc.sff, spc.sff_uncon])
            print(type(sffvals))
            print(sffvals.shape)

            if 'SFF_spectra' not in f.keys():

                f.create_dataset('SFF_spectra', data=sfflist,
                                 maxshape=(None, None))
                f.create_dataset('SFF_spectrum', data=sffvals,
                                 maxshape=(3, None))

                f['SFF_spectra'].attrs['Description'] = sfflist_desc
                f['SFF_spectrum'].attrs['Description'] = sff_desc

            else:
                f['SFF_spectra'].resize(sfflist.shape)
                f['SFF_spectrum'].resize(sffvals.shape)

                f['SFF_spectra'][()] = sfflist
                f['SFF_spectrum'][()] = sffvals

            for key, value in attrs.items():
                f['SFF_spectra'].attrs[key] = value
                f['SFF_spectrum'].attrs[key] = value

    except IndexError:
        pass

    # print(data)
    # files = np.array([np.load(file) for file in files])

    # # create the Spectra class instance

    # taulist = np.logspace(-5, 2, 1000)
    # spc = Spectra(files)
    # spc.spectral_width = (0., 1.)
    # spc.spectral_unfolding(n=3, merge=False, correct_slope=True)
    # spc.get_ham_misc(individual=True)
    # spc.spectral_filtering(filter_key='gaussian', eta=0.5)

    # spc.calc_sff(taulist)

    # sffvals = np.array([spc.taulist, spc.sff, spc.sff_uncon])

    # # ----------------------------------------------------------------------
    # # save the files

    # # folder_path =
    # filename = 'sff_{}_{}'.format(syspar, modpar)
    # print(filename)
    # if not os.path.isdir(savepath):
    #     os.makedirs(savepath)

    # np.save(os.path.join(savepath, filename), sffvals)
