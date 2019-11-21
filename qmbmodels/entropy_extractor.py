#!/usr/bin/env python

"""
A function for extracting entanglement
entropy data from the hdf5 files and store
them into txt files for easier reading,
representation and sharing with others.

"""

import os
import numpy as np
import h5py
import collections

from glob import glob

footer = """
Each row is organised as follows:

dW: disorder strength
average_entropy S: average entropy for a given number of states and samples
rescaled entropy S_re: |log(2) - 2**(2*LA - L - 1) / LA - S/LA|; L-> system,
LA-> subsystem
Delta S: standard deviation of S
size L: system size
nener: number of energies obtained using partial diagonalization
nsamples: number of random samples

"""


def _makehash():
    return collections.defaultdict(_makehash)


def _join(head, tail):

    return os.path.join(head, tail)


def _check(head, tail):

    return os.path.isfile(_join(head, tail))


def _entro_ave(h5file, results_key='Entropy_partial',
               disorder_key='dW'):
    """
    A function that calculates
    the average entropy of the
    data stored in a .hdf5 file
    under a key "Entropy_partial"

    Parameters:
    -----------

    h5file: string
                    Filename of a hdf5 file
                    to be opened.

    results_key: string
                    Which type of results do we want to
                    extract.

    disorder_key: string
                    Which key denotes the disorder
                    parameter in the attributes dict.

    """
    try:
        with h5py.File(h5file, 'r') as file:

            key = results_key
            if key in file.keys():

                entropy = file[key][()]

                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                dW = file[key].attrs['dW']
                size = file[key].attrs['L']
                sub = size / 2.
                ave_entro = -np.mean(entropy)
                entro_rescaled = np.abs(
                    np.log(2) - (2**(2 * sub - size - 1)) / sub -
                    ave_entro / sub)
                std_entro = np.std(entropy)

            else:

                dW = nsamples = nener = ave_entro = std_entro \
                    = size = entro_rescaled = None
                print('Key {} not present in the HDF5 file!'.format(key))

    except IOError:
        print('File {} not present!'.format(h5file))
        dW = nsamples = nener = ave_entro = std_entro \
            = size = entro_rescaled = None

    return dW, ave_entro, entro_rescaled, std_entro, size, nener, nsamples


def _get_r(h5file, results_key='r_data_partial',
           disorder_key='dW'):
    """
    A function that extracts the
    r_values
    data stored in a .hdf5 file
    under the key "r_data_partial"
    or 'r_data'

    Parameters:
    -----------

    h5file: string
                    Filename of a hdf5 file
                    to be opened.

    results_key: string
                    Which type of results do we want to
                    extract.

    disorder_key: string
                    Which key denotes the disorder
                    parameter in the attributes dict.

    """

    try:
        with h5py.File(h5file, 'r') as file:

            key = results_key
            if key in file.keys():

                r_data = file[key][()]

                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                dW = file[key].attrs['dW']
                size = file[key].attrs['L']
                r_val = r_data[1]
                r_err = r_data[2]

            else:

                dW = nsamples = nener = size \
                    = r_val = r_err = None
                print('Key {} not present in the HDF5 file!'.format(key))

    except IOError:
        print('File {} not present!'.format(h5file))
        dW = nsamples = nener = size \
            = r_val = r_err = None

    return dW, r_val, r_err, size, nener, nsamples


# a private dict which specifies which routine should
# be called to perform a desired operation, what
# is the key in the hdf5 dict that returns the
# desired data and what
# is the shape of the return of the associated function.
_routines_dict = {
    'get_entro_ave': [_entro_ave, 'Entropy', 7],
    'get_r': [_get_r, 'r_data', 6]

}


def _crawl_folder_tree(topdir, results_key='Entropy_partial',
                       disorder_key='dW'):
    """
    Crawls the subdirectories of the top results folder
    and provides a list of files to be visited and
    opened for the postprocessing operation to be performed.

    Parameters:
    -----------

    topdir: string,path
                    Should be the path towards the '*/results/'
                    directory where the results for a specific
                    project are stored.

    disorder_key: string



    """
    # filelist = []
    savedict = _makehash()
    # strip any possible underscores
    disorder_key = disorder_key.rstrip('_').lstrip('_')
    if os.path.isdir(topdir):
        # system descriptors, a lisf of
        # subfolders such as
        # 'pbc_True_disorder_uniform_ham_type_spin1d'
        descriptors = os.listdir(topdir)
        descriptors = [desc for desc in descriptors
                       if not _check(topdir, desc)]

        for descriptor in descriptors:

            syspath = os.path.join(topdir, descriptor)
            syspars = os.listdir(syspath)
            syspars = [sysp for sysp in syspars
                       if not _check(syspath, sysp)]

            for syspar in syspars:

                modpath = os.path.join(syspath, syspar)
                modpars = os.listdir(modpath)
                modpars = [modp for modp in modpars
                           if not _check(modpath, modp)]

                for modpar in modpars:
                    savefolder = modpar.split('_{}_'.format(disorder_key))[0]
                    savedict[descriptor][syspar][savefolder] = []

                for modpar in modpars:
                    savefolder, disorder = modpar.split('_{}_'.format(
                        disorder_key))
                    disorder = np.float(disorder)
                    filepath = os.path.join(modpath, modpar)

                    try:
                        file = glob('{}/*.hdf5'.format(filepath))[0]

                        savedict[descriptor][syspar][savefolder].append(
                            [disorder, file])

                    except IndexError:
                        print('file in folder {} not present!'.format(
                            filepath))

    return savedict


def save_ave_entro(topdir, savepath, routine='get_entro_ave',
                   partial=True, disorder_key='dW',
                   footer=footer,
                   savename='entro_sweep'):

    routine = _routines_dict[routine]
    get_fun = routine[0]
    results_key = routine[1]
    if partial:
        results_key += '_partial'
    arr_shape = routine[2]
    savedict = _crawl_folder_tree(
        topdir, disorder_key, results_key=results_key)

    for desc in savedict.keys():

        descdir = os.path.join(savepath, desc)

        for syspar in savedict[desc].keys():

            sysdir = os.path.join(descdir, syspar)

            for savefolder in savedict[desc][syspar].keys():

                savefolder_ = os.path.join(sysdir, savefolder)

                if not os.path.isdir(savefolder_):

                    os.makedirs(savefolder_)

                vals = savedict[desc][syspar][savefolder]

                data = np.zeros((len(vals), arr_shape))

                for i, value in enumerate(vals):

                    data[i] = get_fun(value[1], results_key,
                                      disorder_key)

                # sort according to disorder
                data = data[data[:, 0].argsort()]
                savename = '{}_{}_{}'.format(savename, syspar, savefolder)
                print(_join(savefolder_, savename))
                np.savetxt(_join(savefolder_, savename),
                           data, footer=footer)
