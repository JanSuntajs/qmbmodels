#!/usr/bin/env python

"""
This module provides tools for the calculation
of the ratio of the variances of diagonal
and offdiagonal matrix elements of various
Anderson-type operators. We predominantly
focus on the hopping and local density operators.
As it turns out, those assume a particularly
simple form in the Anderson-type Hamiltonians.

Consider an eigenstate of the Anderson Hamiltonian:

| \alha > = \sum_i \alpha_i | i >,
where | i > is the state in the site-occupational basis
and \alpha_i are the coefficients of the expansion in
that basis.

The matrix element of the density operator between
eigenstates | \alpha > and | \beta > for the site
i reads:

< \alpha | c_i^\dagger c_i | \beta > = \alpha_i^*\beta_i

Similarly, for a hopping operator between two sites,
we get:
< \alpha | c^i_\dagger c_{i+m} + h.c. | \beta > =
= \alpha_i^*\beta_{i+m} + c.c.

https://github.com/JanSuntajs/spectral_statistics_tools


"""

import os
import numpy as np
import glob
import h5py

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.filesaver import save_hdf_datasets, \
    save_external_files, load_eigvals
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general


_hops = ['nn', 'snn']
_matelts_parse_dict = {'matelts_nener': [int, -1]}

_central_name = 'eigvec_components_central_site_partial'
_nn_hop_name = 'eigvec_components_nn_hop_{}_partial'
_snn_hop_name = 'eigvec_components_snn_hop_{}_partial'
_hop_load_files = [_nn_hop_name, _snn_hop_name]

_central_variance_name = 'central_site_matelts_variance_ratio'
_nn_variance_name = 'nn_hop_matelts_variance_ratio'
_snn_variance_name = 'snn_hop_matelts_variance_ratio'
_matelts_mean_values = 'matelts_mean_variances'

_store_files = [_central_variance_name,
                _nn_variance_name, _snn_variance_name,
                _matelts_mean_values]
# sfflist text descriptor


def _format_hop_string(hop_type, dim):

    hop = np.zeros(dim, dtype=np.uint64)
    if hop_type == 'nn':  # nearest neighbour

        hop[0] = 1
    elif hop_type == 'snn':
        if dim == 1:
            hop[0] = 2
        else:
            hop[:2] = 1
    hop_str = '_'.join(str(h_) for h_ in hop)
    return hop_str


def _get_variance_ratio(row1, row2, hop_operator=False):
    """
    This routine is intended for creation
    of various single or two-particle operators
    relevant for the Anderson-type Hamiltonians and
    subsequent calculation of the ratio of variances
    of their diagonal and offdiagonal matrix elements.

    Parameters:

    row1, row2: int
    Indices of rows (of the eigenvector matrix) for which
    to calculate the operator variances

    hop_operator: bool, optional
    Defaults to False. Whether we are calculating for a hop
    operator or a local density operator

    Returns:

    var_ratio: ratio of variances for the sample
    var_diag: variance of the diagonal matrix elements for the sample
    var_offdiag: variance of the offdiagonal matrix elements for the sample
    """

    operator = np.tensordot(row1, np.conj(row2), 0)
    # print(operator.shape)
    if hop_operator:

        operator += np.conj(operator.T)

    # diagonal matrix elements
    diag = operator.diagonal()
    mask_ = ~np.eye(operator.shape[0], dtype=bool)
    # offdiagonal matrix elements
    offdiag = operator[mask_]

    var_diag = np.std(diag)**2  # (np.mean(diag**2) - np.mean(diag)**2)
    # var_offdiag = np.mean(np.abs(offdiag)**2)
    var_offdiag = np.mean(np.abs(offdiag)**2) - np.abs(np.mean(offdiag))**2
    var_ratio = var_diag / var_offdiag

    return np.real(var_ratio), np.real(var_diag), np.real(var_offdiag)


if __name__ == '__main__':

    mateltsDict, matelts_extra = arg_parser_general(_matelts_parse_dict)
    print(mateltsDict)
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    loadpath = savepath.replace('Spectral_stats', 'Eigvals')
    nener = mateltsDict['matelts_nener']
    try:
        file = glob.glob(f"{loadpath}/*.hdf5")[0]
        savelist = []
        with h5py.File(file, 'a', libver='latest', swmr=True) as f:

            # load the central density file
            central_data, attrs, setnames = load_eigvals(
                f, _store_files,
                nener=nener,
                eigname=_central_name.replace('_partial', ''))
            print(setnames)
            print(central_data.shape)

            # find the dimensions of the problem
            dim = attrs['dim']

            # calculate ratio
            # 1st column: ratios for each sample
            # 2nd column: diagonal variances for each sample
            # 3rd column: offdiagonal variances for each sample
            # We calculate two alternate versions of the ratio
            # a) mean over the first column
            # b) mean over 2nd and 3rd column, then take the
            # ratio of the means
            ratiolist_ = np.array([_get_variance_ratio(row, row)
                                  for row in central_data])

            ratiolist = np.mean(ratiolist_[:, 1]) / np.mean(ratiolist_[:, 2])

            savelist.append(ratiolist)

            for i, hopfile in enumerate(_hop_load_files):

                hopfile = hopfile.format(_format_hop_string(_hops[i], dim))
                hopdata, *_ = load_eigvals(f, [hopfile],
                                           nener=mateltsDict['matelts_nener'],
                                           eigname=hopfile.replace('_partial', ''))

                ratiolist_ = np.array([_get_variance_ratio(central_data[i],
                                                           row, True)
                                      for i, row in enumerate(hopdata)])

                ratiolist = np.mean(ratiolist_[:, 1]) / np.mean(ratiolist_[:, 2])
                # 1st column: ratios for each sample
                # 2nd column: diagonal variances for each sample
                # 3rd column: offdiagonal variances for each sample
                savelist.append(ratiolist)

            # take the mean values
            mean_vals = np.zeros((1, 7), dtype=np.float64)
            mean_vals[0][0] = nener
            mean_vals[0][1:4] = [np.mean(ratios) for ratios in savelist]
            mean_vals[0][4:] = [np.std(ratios) for ratios in savelist]
            savelist.append(mean_vals)

            for setname in setnames:
                try:
                    del f[setname]
                except KeyError:
                    print(f'{setname} not present in {file}!')

            save_hdf_datasets({setnames[0]: [savelist[0], (None, )],
                               setnames[1]: [savelist[1], (None,)],
                               setnames[2]: [savelist[2], (None,)],
                               setnames[3]: [savelist[3], (None, 7)]
                               },
                              f, attrs)
            # save txt files for easier reading without the need for
            # inspection of the hdf5 files
        save_external_files(file, {setnames[0]: savelist[0].T,
                                   setnames[1]: savelist[1].T,
                                   setnames[2]: savelist[2].T,
                                   setnames[3]: savelist[3]})

    except IndexError:
        pass

    except KeyError:

        print(('main_matelts_variance_ratio warning'
               ': there was a key error, most'
               ' probably in the .h5py part!'))
