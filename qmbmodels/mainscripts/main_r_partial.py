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


_r_parse_dict = {'r_nener': [int, -1]}
_r_name = 'r_data_partial'
_mean_dist_name = 'r_mean_dist_partial'
_dist_name = 'r_dist_partial'
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

mean_dist_desc = """
This string provides a textual descriptor of the
'r_mean_dist_partial' hdf5 dataset. The latter contains values
of the mean level spacing for each individual disorder
realization/random samples.

The entries are:

r_mean_dist[0, :] -> mean r values flattened into a 1D ndarray

See manuscript at: https://arxiv.org/abs/1905.06345
for a more technical introduction of our implementation.
"""


dist_desc = """
This string provides a textual descriptor of the
'r_dist_partial' hdf5 dataset. The latter contains values
of r statistic for all level spacings and all disorder
samples. The data can be statistically analysed, such
as put in histograms.

The entries are:

r_dist[0, :] -> r values flattened into a 1D ndarray

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
                f, [_r_name, _dist_name, _mean_dist_name], nener=rDict['r_nener'])
            print(setnames)
            attrs.update({'r_desc': r_data_desc})
            attrs.update({'r_mean_dist_desc': mean_dist_desc})
            attrs.update({'r_dist_desc': dist_desc})
            # if the sff spectrum dataset does not yet exist, create it
            spc = Spectra(data)
            spc.spectral_width = (0., 1.)
            gap_data = np.zeros((1, 3))

            # mean r for all samples, std. deviation of gap mean and
            # distribution of mean values
            gap_mean, gap_dev, gap_mean_dist = spc.gap_avg(return_distribution=True)
            gap_data[0] = [data.shape[1], gap_mean, gap_dev]

            gap_dist = spc.gap_dist()

            # except ValueError:
            # pass
            # take care of creation or appending to the hdf5 datasets


            save_hdf_datasets({setnames[0]: [gap_data, (None, 3)],
                setnames[1]: [gap_mean_dist, (1, None)],
                setnames[2]: [gap_dist, (1, None)]}, f, attrs)
            # save txt files for easier reading without the need for
            # inspection of the hdf5 files
        save_external_files(file, {setnames[0]: gap_data,
            setnames[1]: gap_mean_dist.T,
            setnames[2]: gap_dist.T})

    except IndexError:
        pass
