#!/usr/bin/env python

"""
This module provides utilities for determining
the number of degeneracies of the spectrum.

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


_deg_name = 'degeneracies_data'
deg_data_desc = """
This dataset contains a simple measure of the number
of degeneracies (equal energies) present in the spectrum.
The value obtained here is calculated as the sum of
multiplicities of those energy eigenvalues which appear
in the spectrum more than once (and are thus degenerate).
For easier representation, the numbers of degeneracies
for different spectra are averaged in order to obtain
a single scalar value. Before checking for degeneracies,
the values in the spectra are rounded to 14 decimal places
so as to be able to discern actual (physical) degeneracies
from the consequences of different round-offs.
"""


if __name__ == '__main__':

    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            data = f['Eigenvalues'][:]

            attrs = dict(f['Eigenvalues'].attrs)

            spc = Spectra(data)

            degs = np.zeros(spc.nsamples)

            for i, engies in enumerate(spc.spectrum):
                u, c = np.unique(np.around(engies, decimals=12), return_counts=True)
                # degeneracies are all those values which
                # are not unique and hence appear more than once
                degs[i] = np.sum(c[c > 1])
                print(degs[i])

            degs = np.mean(degs, keepdims=True)
            if _deg_name not in f.keys():

                f.create_dataset(_deg_name, data=degs, maxshape=(None))

            else:

                f[_deg_name][()] = degs

            for key, value in attrs.items():
                f[_deg_name].attrs[key] = value

        txt_file = file.replace('eigvals', 'degeneracies')
        txt_file = txt_file.replace('.hdf5', '.txt')
        print(txt_file)
        np.savetxt(txt_file, degs)

    except IndexError:
        pass
