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


_r_name = 'r_data_partial'
# sfflist text descriptor
r_data_desc = """
This string provides a textual descriptor of the
'r_data_partial' hdf5 dataset. The latter contains the
values of the mean ratio of the adjacent level
spacings for an ensemble of hamiltonians for which
the energy spectra have been calculated using partial
diagonalization. The entry corresponding to the
'r_data_partial' key is a ndarray of the shape
(1, 3).

The entries are:

r_data[0, 0] -> number of obtained eigenvalues
r_data[0, 1] -> mean values of the r-statistic for
                the corresponding percentages
r_data[0, 2] -> standard deviations of the mean value

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

        with h5py.File(file, 'a') as f:

            data = f['Eigenvalues_partial'][:]

            attrs = dict(f['Eigenvalues_partial'].attrs)
            attrs.update({'r_desc': r_data_desc})

            # if the sff spectrum dataset does not yet exist, create it
            spc = Spectra(data)
            spc.spectral_width = (0., 1.)
            gap_data = np.zeros((1, 3))

            gap_mean, gap_dev = spc.gap_avg()
            gap_data[0] = [data.shape[1], gap_mean, gap_dev]
            # except ValueError:
            # pass#

            if _r_name not in f.keys():

                f.create_dataset(_r_name, data=gap_data, maxshape=(None, 3))

            else:

                f[_r_name][()] = gap_data

            for key, value in attrs.items():
                f[_r_name].attrs[key] = value

        txt_file = file.replace('eigvals', 'r_stats_partial')
        txt_file = txt_file.replace('.hdf5', '.txt')
        print(txt_file)
        np.savetxt(txt_file, gap_data)

    except IndexError:
        pass
