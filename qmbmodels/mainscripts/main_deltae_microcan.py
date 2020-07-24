#!/usr/bin/env python

"""
This module calculates the average mean level spacing
of an ensemble of spectra that were obtained
using partial diagonalizaton. We also refer to the
calculations performed here as the calculation
of the microcanonical mean level spacing since the
spectral widths in question here are usually significantly
smaller than the ones in the full spectra.

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

_e_parse_dict = {'r_nener': [int, -1], }
_deltaE_name = 'deltaE_data_partial'
# sfflist text descriptor
deltaE_data_desc = """
This string provides a textual descriptor of the
'deltaE_data_partial' hdf5 dataset. The latter contains
the values related to the (trivial) calculation of the
spectral width.

The entries are:

deltaE_data[0, 0] -> local Hilbert space dimension
deltaE_data[0, 1] -> number of spectra in the ensemble
deltaE_data[0, 2] -> Mean level spacing averaged across
                     all the spectra in the ensemble
deltaE_data[0, 3] -> Standard deviation of the mean level
                     spacing calculated above.
deltaE_data[0, 4] -> Mean width of the spectra averaged
                     across all the spectra in the ensemble.
deltaE_data[0, 5] -> Standard deviation of width of the spectra
                     calculated under the previous entry.

Since the widths of the spectra are calculated for a relatively
narrow energy span compared to the full width of the said spectra,
we also refer to the calculations as the 'calculations in the
microcanonical ensemble.'
"""


if __name__ == '__main__':

    # rDict, rextra = arg_parser_general(_r_parse_dict)
    eDict, rextra = arg_parser_general(_e_parse_dict)
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            data, attrs, setnames = load_eigvals(f, [_deltaE_name],
                                                 nener=eDict['r_nener'])

            hilbert_dim = data.shape[1]
            attrs = dict(f['Eigenvalues_partial'].attrs)
            attrs.update({'deltaE_desc': deltaE_data_desc})

            # if the sff spectrum dataset does not yet exist, create it
            widths = data[:, -1] - data[:, 0]
            mean_width = np.mean(widths)
            std_width = np.std(widths)
            # mean level spacing
            mean_lvl_spc = mean_width / (hilbert_dim * 1.0)
            std_lvl_spc = np.std(widths / (hilbert_dim * 1.0))

            deltaE_data = np.zeros((1, 6))
            deltaE_data[0] = [hilbert_dim, data.shape[0], mean_lvl_spc,
                              std_lvl_spc,
                              mean_width, std_width]
            # except ValueError:
            # pass#
            save_hdf_datasets(
                {setnames[0]: [deltaE_data, (None, 6)]}, f, attrs)

        save_external_files(file, {setnames[0]: deltaE_data})
    except IndexError:
        pass
