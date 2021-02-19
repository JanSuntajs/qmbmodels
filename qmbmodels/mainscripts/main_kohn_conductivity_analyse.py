#!/usr/bin/env python

"""


"""

import os
import numpy as np
import glob
import h5py
import math

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.filesaver import save_hdf_datasets, \
    save_external_files, load_eigvals
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general


_kohn_parse_dict = {'kohn_nener': [int, -1]}
_kohn_name = 'kohn_data'
_mean_dist_name = 'kohn_mean_dist'
_thou_dist_name = 'spectral_differences_dist'
_heis_dist_name = 'level_spacings_dist'
# sfflist text descriptor
kohn_data_desc = """
This string provides a textual
description of the kohn_data
hdf5 dataset.

Note that whenever we discuss the Thouless
Energy, we refer to the differece of spectra after
slightly perturbing the boundary conditions with a
small complex phase. To obtain the Thouless energy
shown here, we take the absolute value of the differences
and rescale them by 2/\phi**2, where \phi is the said
complex phase.
The entry corresponding
to the 'kohn_data' key is a nd array
of the shape (1, 10):

The entries are:
kohn_data[0,0] -> phase_factor \phi
kohn_data[0, 1] -> nener_orig -> the
original number of energies, equal to the
dimension of the Hilbert space
kohn_data[0, 2] -> nener, the number of energies
used in the thouless time extraction
kohn_data[0, 3] -> g_1 -> "Kohn" conductivity, 
calculated by obtaining the ratio of each samples'
Thouless and Heisenberg energy and then taking the
average of the ratios over different disorder realizations.
kohn_data[0, 4] -> g_2 -> "Kohn" conductivity, calculated
by taking the ratio of the Thouless and Heisenberg energy,
where each of those have been calculated *globally*, that is,
by taking their mean over all spectra and disorder realizations
before taking their ratio.
kohn_data[0, 5] -> Thouless energy, obtained *globally*, hence
by taking the average value across different energy spectra
and different disorder realizations. The quantity being averaged
is the absolute value of the difference of spectra upon changing
the boundary conditions, rescaled by 2/\phi**2. The averaging
over energies is over nener values.
kohn_data[0, 6] -> Heisenberg energy, obtained *globally.* The averaging
over energies is over nener_orig values.
kohn_data[0, 7] -> standard deviation of g_1 -> data under kohn_data[0,3]
kohn_data[0, 8] -> standard deviation of kohn_data[0, 5]
kohn_data[0, 9] -> standard deviation of kohn_data[0, 6]
"""

mean_dist_desc = """
The description of the "kohn_mean_dist" hdf5 dataset.
Distributions of mean values of the Thouless energies (curvatures)
and Heisenberg energies over different disorder realizations.
As noted, Thouless energies are obtained by taking a difference
of two spectra upon slightly changing the boundary conditions
(introducing a small phase factor to them), then taking the absolute
value of the differences and rescaling them by 2/\phi**2, where
\phi is the said phase.

kohn_mean_dist[0] -> distribution of mean values for
the thouless energies across different disorder realizations.

kohn_mean_dist[1] -> distribution of mean values for the
Heisenberg energies of different spectra (for different
disorder realizations.)

"""


thou_dist_desc = """
The description of the "spectral_differences_dist" hdf5 dataset.
Spectral differences (their absolute value rescaled by
2/\phi**2, where \phi is the phase. In literature, this
is usually denoted as "curvatures") across different
spectra and different energies. This form is appropriate
for showing histograms.
"""

heis_dist_desc = """
Distribution of spacings between levels in the unperturbed system.
"""


if __name__ == '__main__':

    kohnDict, kohnextra = arg_parser_general(_kohn_parse_dict)
    print(kohnDict)
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')

    nener = kohnDict['kohn_nener']
    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]

        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            # load the appropriate eigenvalue files
            # always load all the energies
            data, attrs, setnames = load_eigvals(
                f, [_kohn_name, _mean_dist_name, _thou_dist_name,
                    _heis_dist_name],
                nener=-1,
                partial=False)
            print(setnames)

            # check if spectra with changed bc are present
            phase_factor = attrs['phase_bc']
            key_diffs = 'Spectrum_differences_phase_factor'
            # if key_diffs is present in attributes' key,
            # then we can perform our calculation
            diffs_present = [key for key in f.keys()
                             if key_diffs in key]

            # proceed with the calculation
            if any(diffs_present):
                key_diffs = diffs_present[0]

                data_phase = f[key_diffs][:]
                # rescale the data_phase values
                # we are calculating differences
                # of spectra once the boundary
                # conditions are changed. What is
                # usually studied in the literature
                # is the *curvature*, which is rescaled
                # as shown here

                # slice the data if not all energies
                # are needed
                nener_orig = attrs['nener']
                if nener != -1:
                    if nener > nener_orig:
                        nener = nener_orig

                    remain = nener_orig - nener
                    start = math.ceil(remain / 2.)
                    stop = start + nener
                    data_phase = data_phase[:, slice(start, stop)]

                    for i in range(len(setnames) - 1):
                        setnames[i] += f'_nev_{nener}'

                data_phase *= 2. / phase_factor**2
                data_phase = np.abs(data_phase)

                # calculations for the Heisenberg
                # energy are done on the unperturbed
                # spectra
                # calculate two types of the Heisenberg
                # energy:
                # type 1: get E_H for each spectrum
                # type 2: get E_H as a global mean
                # across all values (both over spectra)
                # and disorder realizations)
                level_spacings = np.diff(data)
                # this is an array -> one value for each
                # disorder realization
                e_heis_1 = np.mean(level_spacings, axis=1)
                # this is a scalar -> a global mean
                e_heis_2 = np.mean(e_heis_1)
                e_heis_2_std = np.std(e_heis_1)

                # we get the conductance in two ways as well
                # type 1: we take the ratio of the Thouless
                # energy and heisenberg energy for each
                # disorder realization and then average over
                # different disorder realizations.
                # type 2: we take the thouless energy and
                # heisenberg energy globally and then take
                # their ratio

                e_thou_1 = np.mean(data_phase, axis=1)
                e_thou_2 = np.mean(e_thou_1)
                e_thou_2_std = np.std(e_thou_1)

                # take the two ratios

                g_1 = np.mean(e_thou_1 / e_heis_1)
                g_1_std = np.std(e_thou_1 / e_heis_1)
                g_2 = e_heis_2 / e_thou_2

                # ----------------------
                # kohn_mean values
                # ----------------------
                attrs.update({'kohn_desc': kohn_data_desc,
                              'kohn_mean_dist_desc': mean_dist_desc,
                              'kohn_thou_dist_desc': thou_dist_desc,
                              'kohn_heis_dist_desc': heis_dist_desc,
                              'nener0': nener_orig,
                              'nener': nener})

                kohn_data = np.zeros((1, 10))
                kohn_data[0] = [phase_factor, nener_orig, nener,
                                g_1, g_2, e_thou_2, e_heis_2,
                                g_1_std, e_thou_2_std, e_heis_2_std]

                # distributions
                mean_dist_data = np.array([e_thou_1,
                                           e_heis_1])
                # take care of creation or appending to the hdf5 datasets
                for setname in setnames:
                    try:
                        del f[setname]
                    except KeyError:
                        print(f'{setname} not present in {file}!')

                # 0: kohn_data
                # 1: dist of thouless and heiss energies for diff.
                # disorders
                # 2:
                # 3:
                save_hdf_datasets({setnames[0]: [kohn_data, (None, 10)],
                                   setnames[1]: [mean_dist_data,
                                                 (2, None)],
                                   setnames[2]: [data_phase.flatten(),
                                                 (None,)],
                                   setnames[3]: [level_spacings.flatten(),
                                                 (None,)]
                                   },
                                  f, attrs)
                # save txt files for easier reading without the need for
                # inspection of the hdf5 files
                save_external_files(file, {setnames[0]: kohn_data,
                                           setnames[1]: mean_dist_data.T,
                                           setnames[2]: data_phase.flatten().T,
                                           setnames[3]:
                                           level_spacings.flatten().T})

    except IndexError:
        pass
