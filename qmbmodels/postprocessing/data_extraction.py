import os
import numpy as np
from glob import glob
from collections import defaultdict

from .available_routines import _routines_dict
from . import utils as ut
# some helper routines


def _makehash():
    return defaultdict(_makehash)


def _join(head, tail):

    return os.path.join(head, tail)


def _check(head, tail):

    return os.path.isfile(_join(head, tail))

# a routine for extracting the part of the filename
# describing the disorder key and the corresponding
# value of the disorder parameter


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

    Returns:

    savedict: dict
        A dictionary of the tree-like folder structure containing
        the files (filenames) which store the actual numerical
        results. The form of the dictionary is as follows:
        savedict[descriptor][syspar][savefolder]

        Where we have:
        descriptor: string containing the basic description
        of the system studies, such as:
        'pbc_True_disorder_uniform_ham_type_spin1d'

        syspar: string describing the system parameters, for
        instance:
        'L_10_nu_5'

        savefolder: a string describing model parameters without
        the part containing the disorder key and its corresponding
        disorder parameter value. If the model parameter descriptor
        were, for instance:
        'J1_1.0_J2_1.0_delta1_0.55_delta2_0.55_W_0.0_dW_1'
        and the disorder key were 'dW', the savefolder string would
        be:
        'J1_1.0_J2_1.0_delta1_0.55_delta2_0.55_W_0.0'

        An entry of such a nested dictionary is a list of lists where
        each nested list has the folowing structure:
        [<disorder_parameter_value>, <path_to_the_file_with_results>]

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

            syspath = _join(topdir, descriptor)
            syspars = os.listdir(syspath)
            syspars = [sysp for sysp in syspars
                       if not _check(syspath, sysp)]

            for syspar in syspars:

                modpath = _join(syspath, syspar)
                modpars = os.listdir(modpath)
                modpars = [modp for modp in modpars
                           if not _check(modpath, modp)]

                for modpar in modpars:
                    savefolder, tmp = ut._extract_disorder(
                        modpar, disorder_key)
                    print(savefolder)
                    savedict[descriptor][syspar][savefolder] = []

                for modpar in modpars:
                    savefolder, disorder = ut._extract_disorder(
                        modpar, disorder_key)
                    filepath = _join(modpath, modpar)
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
                 savename='entro_sweep',
                 reverse_order=False,
                 collapse=False,
                 exclude_keys=[],
                 *args, **kwargs):
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

    # in case we wish to exclude some keys from the dictionary
    ut.delete_keys_from_dict(savedict, exclude_keys)

    # if we have the reversed saving order -> first the
    # module_parameters bar the disorder part,
    # then the system parameters
    if reverse_order:
        savedict_flipped = _makehash()
        # key: desc, val: dict with all else
        for key, val in savedict.items():
                # subkey1: syspar, subval1: dict with all else
            for subkey1, subval1 in val.items():
                # subkey2:
                for subkey2, subval2 in subval1.items():

                    savedict_flipped[key][subkey2][subkey1] = subval2

        savedict = savedict_flipped

    # iterate over description keys; an example
    # of a descriptor key would be:
    # 'pbc_True_disorder_uniform_ham_type_spin1d'
    for desc in savedict.keys():

        descdir = os.path.join(savepath, desc)

        for syspar in savedict[desc].keys():

            sysdir = _join(descdir, syspar)

            for savefolder in savedict[desc][syspar].keys():

                if (reverse_order and collapse):
                    savefolder_ = sysdir
                else:
                    savefolder_ = _join(sysdir, savefolder)

                if not os.path.isdir(savefolder_):

                    os.makedirs(savefolder_)

                vals = savedict[desc][syspar][savefolder]
                print(vals)
                data = np.zeros((len(vals), arr_shape), dtype=np.float64)

                for i, value in enumerate(vals):

                    data[i] = get_fun(value[1], results_key,
                                      disorder_key, *args, **kwargs)

                # sort according to disorder
                data = data[data[:, 0].argsort()]

                names = (syspar, savefolder)
                if reverse_order:
                    names = names[::-1]
                savename_ = '{}_{}_{}'.format(savename, *names)
                print(_join(savefolder_, savename_))
                np.savetxt(_join(savefolder_, savename_),
                           data, footer=footer)
