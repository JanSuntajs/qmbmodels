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

footer_entro_post = """
Each row is organised as follows:

dW: disorder strength
average_entropy S: average entropy for a given number of states and samples
rescaled entropy S_re: |log(2) - 2**(2*LA - L - 1) / LA - S/LA|; L-> system,
LA-> subsystem
Delta S: standard deviation of S
size L: system size
nener: number of energies obtained using partial diagonalization
nsamples: number of all the random samples
nsamples_selected: number of the random disorder samples with an appropriate
variance
nsamples_selected: nsamples - nsamples_selected
population_variance: the theoretical prediction for the variance of the
disorder distribution
epsilon: condition used to determine whether to select a given disorder
distribution:
|\sigma^2_sample - \sigma_^2_theory| < epsilon * pop_var / sqrt(size - 1)
"""


footer_entro = """
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

footer_r = """
Each row is organised as follows:

dW: disorder strength
r_val r: mean ratio of the level spacings
delta r_val Delta r: standard deviation of the r value
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


def _entro_ave_postprocessed(h5file, results_key, disorder_key='dW',
                             var_factor=1. / 3., limit_disorder=2.,
                             mu=1.5):
    """
    Since we are dealing with finite-size samples away from the
    thermodynamic limit, we need to account for that in our
    postprocessing routines. Our Hamiltonians are typically
    prepared using random potentials where the potential values
    are drawn from some probability distribution with a given
    set of parameters (e.g. mean, variance, etc.). Due to the
    finite size of our samples, the sample means and variances
    might differ from those of the whole population. In fact, as
    it is known from the probability and statistics, the sample
    means and variances are distributed according to appropriate
    probability distributions. Since we only wish to analyze results
    corresponding to samples whose variances do not deviate too much
    from the desired population variance, we devise a routine based on
    which we select or reject calculation results to be included
    in our subsequent analysis.

    The selection criterion is as follows:

    |\sigma^2_sample - \sigma^2_pop | <  \epsilon

    Where we calculate epsilon according to the following rule:

    \epsilon = \frac{1}{\sqrt{L - 1}} * \sqrt(mu) * limit_disorder^2 * \
    var_factor

    Here, \sigma^2_sample is the samples variance in the unbiased form,
    \sigma^2_pop is the population variance of the selected probability dist.

    L is the length of the 1-dimensional quantum chain.

    \mu is a numerical prefactor which we set to the value of 1.5 for the
    box distribution and equals 2 for the uniform distribution. Note that
    the value for the box distribution was deduced experimentally and not
    in any way rigorously.

    limit_disorder sets the value of disorder agains which we wish to
    compare/test our results.

    var_factor is a multiplicative factor used to obtain the population
    variance from the disorder parameter value. In general:
    \sigma^2_pop = var_factor * disorder^2
    For box distribution, var_factor = 1./3.



    """

    disorder_string = 'Hamiltonian_random_disorder_partial'

    dW = nsamples = nener = ave_entro = std_entro \
        = size = entro_rescaled = nsamples_rejected \
        = nsamples_selected = None

    try:
        with h5py.File(h5file, 'r') as file:

            key = results_key

            if ((disorder_string in file.keys()) and (key in file.keys()
                                                      )):

                disorder = file[disorder_string][()]

                nsamples = file[disorder_string].attrs['nsamples']
                nener = file[key].attrs['nener']
                size = file[disorder_string].attrs['L']
                dW = np.float(file[key].attrs[disorder_key])

                # check if there is the same number of disorder samples
                # as there are spectra
                check_shapes = (nsamples == disorder.shape[0])

                if not check_shapes:

                    # ------------------------------------------------
                    #
                    #  Calculation of variances of the disorder distribution
                    #
                    # ------------------------------------------------
                    # get the variances of disorder and then reject the
                    # inappropriate samples
                    # variances of the disordered distribution
                    # samples
                    variances = np.var(disorder, axis=1, ddof=1)
                    population_variance = dW**2 * var_factor

                    # the selection/rejection criterion
                    epsilon = np.sqrt(mu) * var_factor * \
                        limit_disorder**2 / np.sqrt(size - 1)

                    condition = np.abs(
                        variances - population_variance) < epsilon
                    nsamples_selected = np.sum(condition)

                    entropy = file[key][()]
                    entropy = entropy[condition]
                    sub = size / 2.
                    ave_entro = np.mean(entropy)
                    entro_rescaled = np.abs(
                        np.log(2) - (2**(2 * sub - size - 1)) / sub -
                        ave_entro / sub)
                    std_entro = np.std(entropy)

                    nsamples_rejected = nsamples - nsamples_selected
                else:

                    print('Shape mismatch!')

            else:

                print('Key {} or {} not present in the HDF5 file!'.format(
                    key, disorder_string))
    except IOError:
        print('File {} not present!'.format(h5file))

    return (dW, ave_entro, entro_rescaled, std_entro, size, nener, nsamples,
            nsamples_selected, nsamples_rejected, population_variance, epsilon)


def _entro_ave(h5file, results_key, disorder_key='dW'):
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
                dW = np.float(file[key].attrs[disorder_key])
                size = file[key].attrs['L']
                sub = size / 2.
                ave_entro = np.mean(entropy)
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


def _get_r(h5file, results_key,
           disorder_key='dW', full=False, target_percentage=None):
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

    full: boolean
          Whether the full or partial diagonalization
          results are being extracted.

    target_percentage: float, None
          If full == True, which percentage of states
          is to be selected.

    """

    try:
        with h5py.File(h5file, 'r') as file:

            key = results_key
            if key in file.keys():

                r_data = file[key][()][0]

                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                dW = np.float(file[key].attrs[disorder_key])
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
    'get_entro_ave_post': [_entro_ave_postprocessed, 'Entropy', 11,
                           footer_entro_post],
    'get_entro_ave': [_entro_ave, 'Entropy', 7, footer_entro],
    'get_r': [_get_r, 'r_data', 6, footer_r]

}


def _extract_disorder(string, disorder_key):
    """
    An internal routine for returning the
    disorder key and the rest of the input
    string
    """

    # append '_' at the beginning of the
    # string to make splitting w.r.t. the
    # disorder_key easier
    string = '_' + string

    # make sure there are no trailing or preceeding
    # multiple underscore by removing them
    disorder_key = disorder_key.lstrip('_').rstrip('_')
    # now make sure there is exactly one trailing
    # and one preceeding underscore
    disorder_key = '_{}_'.format(disorder_key)
    # split w.r.t. the disorder_key. The first
    # part does not contain the disorder parameter
    # value, while the second one does
    rest1, dis_string = string.split(disorder_key)
    # find the first occurence of '_' in the
    # dis_string, which indicates the length of
    # the disorder parameter value
    splitter = dis_string.find('_')
    if splitter < 0:
        disorder = dis_string
        rest2 = ''
    else:
        disorder, rest2 = dis_string[:splitter], dis_string[splitter:]

    disorder = np.float(disorder)

    # the part without the disorder value
    rest = rest1.lstrip('_') + rest2
    return rest, disorder


def _crawl_folder_tree(topdir, results_key,
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
                    savefolder, tmp = _extract_disorder(modpar, disorder_key)
                    print(savefolder)
                    savedict[descriptor][syspar][savefolder] = []

                for modpar in modpars:
                    savefolder, disorder = _extract_disorder(
                        modpar, disorder_key)
                    filepath = os.path.join(modpath, modpar)
                    print('filepath')
                    print(filepath)
                    try:
                        file = glob('{}/*.hdf5'.format(filepath))[0]

                        savedict[descriptor][syspar][savefolder].append(
                            [disorder, file])

                    except IndexError:
                        print('file in folder {} not present!'.format(
                            filepath))

    return savedict


def extract_data(topdir, savepath, routine='get_entro_ave',
                 partial=True, disorder_key='dW',
                 savename='entro_sweep'):
    """

    """

    routine = _routines_dict[routine]
    get_fun = routine[0]
    results_key = routine[1]
    footer = routine[3]
    if partial:
        results_key += '_partial'
    arr_shape = routine[2]
    savedict = _crawl_folder_tree(
        topdir, results_key, disorder_key=disorder_key)

    for desc in savedict.keys():

        descdir = os.path.join(savepath, desc)

        for syspar in savedict[desc].keys():

            sysdir = os.path.join(descdir, syspar)

            for savefolder in savedict[desc][syspar].keys():

                savefolder_ = os.path.join(sysdir, savefolder)

                if not os.path.isdir(savefolder_):

                    os.makedirs(savefolder_)

                vals = savedict[desc][syspar][savefolder]
                print(vals)
                data = np.zeros((len(vals), arr_shape), dtype=np.float64)

                for i, value in enumerate(vals):

                    data[i] = get_fun(value[1], results_key,
                                      disorder_key)

                # sort according to disorder
                data = data[data[:, 0].argsort()]
                savename_ = '{}_{}_{}'.format(savename, syspar, savefolder)
                print(_join(savefolder_, savename_))
                np.savetxt(_join(savefolder_, savename_),
                           data, footer=footer)
