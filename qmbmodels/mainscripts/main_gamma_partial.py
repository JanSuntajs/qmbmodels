#!/usr/bin/env python

"""
This module provides utilities for calculating
the gamma value for the calculated spectra.
Gamma value is calculated as:

Gamma**2 = Tr(H**2)/D - Tr(H)**2/D**2
Where D is the Hilbert space dimension.

This implementation uses tools from the
spectral_statistics_tools package
available at:

https://github.com/JanSuntajs/spectral_statistics_tools


"""

import os
import numpy as np
import glob
import h5py


from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general


_gamma_name = 'gamma_data_partial'
# sfflist text descriptor
gamma_data_desc = """
This string provides a textual descriptor of the
'gamma_data_partial' hdf5 dataset. The latter contains the
values related to the calculation of the Gamma spectral
observable, which is calculated as follows:

Gamma**2 = Tr(H**2)/D - Tr(H)**2/D**2
Where H is the Hamiltonian and D is the Hilbert space
dimension.

The entries are:

gamma_data[0, 0] -> Gamma value
gamma_data[0, 1] -> Gamma**2
gamma_data[0, 2] -> Standard deviation of Gamma**2 (calculated
                    using numpy std() function)
gamma_data[0, 3] -> D (Hilbert space dimension)
gamma_data[0, 4] -> Mean value of Tr(H) averaged over an ensemble
                    of Hamiltonians.
gamma_data[0, 5] -> Mean value of Tr(H**2) averaged over an ensemble
                    of Hamiltonians.

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of our implementation.
"""


if __name__ == '__main__':

    # rDict, rextra = arg_parser_general(_r_parse_dict)
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            # Diagonal matrix elements of the Hamiltonian:
            matelts = f['Hamiltonian_diagonal_matelts_partial'][:]
            # Hilbert space dimension
            hilbert_dim = matelts.shape[1]
            # Diagonal matrix elements of the squared hamiltonian (H**2)
            matelts_sq = f['Hamiltonian_squared_diagonal_matelts_partial'][:]

            matelts_sq_ = np.sum(matelts_sq, axis=1)
            matelts_ = np.sum(matelts, axis=1)
            gamma_sq_ = matelts_sq_ / hilbert_dim - \
                (np.sum(matelts_, axis=1) / hilbert_dim)**2

            gamma_sq = np.mean(gamma_sq_)
            gamma_sq_dev = np.std(gamma_sq_)

            attrs = dict(f['Hamiltonian_diagonal_matelts_partial'].attrs)
            attrs.update({'gamma_desc': gamma_data_desc})

            # if the sff spectrum dataset does not yet exist, create it

            gamma_data = np.zeros((1, 6))

            gamma_data[0] = [np.sqrt(gamma_sq), gamma_sq,
                             gamma_sq_dev, hilbert_dim,
                             np.mean(matelts_), np.mean(matelts_sq_)]
            # except ValueError:
            # pass#

            if _gamma_name not in f.keys():

                f.create_dataset(_gamma_name, data=gamma_data,
                                 maxshape=(None, 6))

            else:

                f[_gamma_name][()] = gamma_data

            for key, value in attrs.items():
                f[_gamma_name].attrs[key] = value

        txt_file = file.replace('eigvals', 'gamma_stats_partial')
        txt_file = txt_file.replace('.hdf5', '.txt')
        print(txt_file)
        np.savetxt(txt_file, gamma_data)

    except IndexError:
        pass
