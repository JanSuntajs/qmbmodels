#!/usr/bin/env python

"""
This module provides utilities for calculating
the mean ratio of the adjacent level spacings.
The calculation is performed for different
percentages of states from around the middle
of the spectrum.

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

from utils import set_mkl_lib
from utils.cmd_parser_tools import arg_parser, arg_parser_general

_r_keys = ['r_step']

_r_parse_dict = {'r_step': [float, 0.25], }

# sfflist text descriptor
r_data_desc = """
This string provides a textual descriptor of the
'r_data' hdf5 dataset. The latter contains the
values of the mean ratio of the adjacent level
spacings for an ensemble of hamiltonians for which
the energy spectra have been calculated using exact
diagonalization. The entry corresponding to the
'r_data' key is a ndarray of the shape
(n_data, 3). Here, n_data is the number of different
percentages of the eigenstates for which we calculate
the r-statistic. As it is customary with this observable,
we usually only consider some portion of the states
from around the centre of the energy spectrum.

The entries are:

r_data[:, 0] -> percentages of the states for which
                the values have been calculated
r_data[:, 1] -> mean values of the r-statistic for
                the corresponding percentages
r_data[:, 2] -> standard deviations of the mean values

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of our implementation.
"""


if __name__ == '__main__':

    rDict, rextra = arg_parser_general(_r_parse_dict)
    argsDict, extra = arg_parser([], [])

    r_step, = [
        rDict[key] for key in _r_parse_dict.keys()]

    print(r_step)

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a') as f:

            data = f['Eigenvalues'][:]

            attrs = dict(f['Eigenvalues'].attrs)
            attrs.update(rDict)

            # if the sff spectrum dataset does not yet exist, create it
            n_data = int(0.5 / r_step)
            gap_data = np.zeros((n_data, 3))
            spc = Spectra(data)

            # try:
            for i in range(int(0.5 / r_step)):
                percentage = 1 - (2 * i * r_step)
                spc.spectral_width = (i * r_step, 1 - i * r_step)

                gap_mean, gap_dev = spc.gap_avg()
                gap_data[i] = [percentage, gap_mean, gap_dev]
            # except ValueError:
                # pass#

            if 'r_data' not in f.keys():

                f.create_dataset('r_data', data=gap_data, maxshape=(None, 3))

            else:
                f['r_data'].resize(gap_data.shape)

                f['r_data'][()] = gap_data

            for key, value in attrs.items():
                f['r_data'].attrs[key] = value

        txt_file = file.replace('eigvals', 'r_stats')
        txt_file = txt_file.replace('.hdf5', '.txt')
        print(txt_file)
        np.savetxt(txt_file, gap_data)

    except IndexError:
        pass
