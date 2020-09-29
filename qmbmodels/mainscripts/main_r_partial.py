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

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.filesaver import save_hdf_datasets, \
    save_external_files, load_eigvals
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general


_r_parse_dict = {'r_nener': [int, -1], 
                 'r_bins': [int, 100]}
_r_name = 'r_data_partial'
_hist_name = 'r_hist_partial'
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

r_hist_desc = """
This string provides a textual descriptor of the
'r_hist_partial' hdf5 dataset. The latter contains
the histogramed values of the level spacings averaged
over different random spectra.
(2, nbins).

The entries are:

r_hist[0, 0] -> histogram edges
r_hist[0, 1] -> histogram values.


See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of our implementation.
"""


if __name__ == '__main__':

    rDict, rextra = arg_parser_general(_r_parse_dict)
    print(rDict)
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            # load the appropriate eigenvalue files
            data, attrs, setnames = load_eigvals(
                f, [_r_name, _hist_name], nener=rDict['r_nener'])
            print(setnames)
            attrs.update({'r_desc': r_data_desc})
            attrs.update({'r_hist_desc': r_hist_desc})
            # if the sff spectrum dataset does not yet exist, create it
            spc = Spectra(data)
            spc.spectral_width = (0., 1.)
            gap_data = np.zeros((1, 3))

            gap_mean, gap_dev = spc.gap_avg()
            gap_data[0] = [data.shape[1], gap_mean, gap_dev]

            bins = rDict['r_bins']
            gap_hist_data = np.zeros((2, bins+1))
            gap_hist, gap_edges = spc.gap_hist(bins=bins,
                **{'cumulative': True})
            gap_hist_data[0,:] = gap_edges
            gap_hist_data[1,:-1] = gap_hist
            gap_hist_data[1,-1] = 1.
            # except ValueError:
            # pass
            # take care of creation or appending to the hdf5 datasets
            save_hdf_datasets({setnames[0]: [gap_data, (None, 3)],
                setnames[1]: [gap_hist_data, (2, None)]}, f, attrs)
            # save txt files for easier reading without the need for
            # inspection of the hdf5 files
        save_external_files(file, {setnames[0]: gap_data,
            setnames[1]: gap_hist_data.T})

    except IndexError:
        pass
