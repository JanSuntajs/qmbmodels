"""
This module implements a routine for extraction of
the relevant data from a selected folder.

Given the main folder with the calculation results
and the requested type of the results, the
extract_data(...) routine allows one to first find
all the results stored in the main folder's subfolders,
then load the requested data to disk, perform some
post-processing analysis and then store the results
to the appropriate folder.

"""


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
                 merge=True,
                 append_to_results_key='',
                 *args, **kwargs):
    """
    A routine for extracting numerical
    data all at once, performing
    some kind of statistical analysis on those
    and then saving the results. Especially
    useful for performing a postprocessing
    analysis of parameter sweeps. The routine works
    on sets of data stored in directories with
    a predefined folder structure:
        <topdir>/<desc>/<syspar>/<modpar>
        An example would be:
        <topdir> = /somepath/ (path to where all
        the results for a given task are stored)
        <desc> = pbc_True_disorder_uniform_ham_type_spin1d
        (a basic description of the model, containing
        info on the boundary conditions, type of disorder,
        type of the hamiltonian ...)
        <syspar> = L_10_nu_5
        (info such as the system size, number of up electrons
        etc.)
        <modpar> = J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0_dW_2.0
        (values of the different Hamiltonian parameters)
    This function crawls all the subdirectories of the topdir looking
    for the .hdf5 files storing the results which can then be loaded
    and analysed. Omitting some results from the consideration
    is also possible as it shall be described below.

    Parameters:
    -----------

    topdir: string
    Path to the top folder of our data storage
    directory.

    savepath: string
    Path to the directory where the data should
    be stored. If the folder does not yet exist,
    the helper routines within this function
    take care of its creation.

    routine: string
    Which routine to call for postprocessing.
    Available routines are listed in
    .available_routines._routines_dict dictionary.
    Example:
    'get_entro_ave' would initiate the
    .entropy.entro_ave(...) function.

    partial: boolean
    Whether the results for the partial diagonalization
    case are to be processed.

    disorder_key: string
    A string specifying which part of the module parameter
    string corresponds to the disorder strength parameter.
    Example: 'dW'
    If we have, for example, the following modpar string:
    'J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0_dW_2.0',
    specifying the above disorder_string would allow us
    to extract the value of 2.0 of the disorder strength
    parameter. On top of that, the underlying routines
    would also extract the sweep substring
    'J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0'
    containing the values of all the parameters that
    remain constant during the disorder parameter sweep.

    savename: string
    A string to be prepended to the name of the saved
    results file which is otherwise of the fixed form
    where the name is a combination of syspar string
    and the modpar string without the disorder descriptor
    part.
    Example: 'entro_W_sweep'
    For a system with modpar string 'L_12_nu_6'
    (a system of size 12 with 6 up electrons)
    and with a sweep substring given above, the
    standard filename form would be
    'L_12_nu_6_J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0'
    Once the savename is appended, the full name would
    thus be:
    'entro_W_sweep_L_12_nu_6_J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0'

    reverse_order: boolean
    Whether to reverse order of the folders when saving files.
    If False, the corresponding directory structure (with
    a practical example) should be as follows:

    <topdir>/<desc>/<syspar>/<sweep_substring>/ -> results

    If the results are stored under /somepath/, an
    example structure would be:

    /somepath/pbc_True_disorder_uniform_ham_type_spin1d/L_10_nu_5/
    J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0

    Since finite-size scaling analysis is often performed on sets
    of data generated by this function, it may be preferable to have
    all the results for different system sizes and the same
    set of model parameters in one place. Setting reverse_order
    to True would thus yield the following directory structure:

    <topdir>/<desc>/<sweep_substring>/<syspar>/ -> results

    /somepath/pbc_True_disorder_uniform_ham_type_spin1d/
    J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0/L_10_nu_5

    collapse: boolean
    Defaults to False. If True, it only takes effect if
    also reverse_order==True. Prevents the creation of
    the syspar directory and so the resulting folder
    structure is as follows:
    <topdir>/<desc>/<sweep_substring>
    Hence the results for different system sizes are
    all stored in the same folder.

    exclude_keys: list
    Allows users to specify if the results from some
    (sub)folders within the topdir are to be excluded
    from the analysis.

    merge: boolean
    Whether to merge the results for different values of
    the sweeping parameter in one file. If False, the
    results for different values of sweeping parameters
    are stored into different files.

    *args, **kwargs: arguments to be passed to the actual
    postprocessing routine.

    Returns:
    --------

    This function has no return, it only stores the results.
    """

    routine = _routines_dict[routine]
    get_fun = routine[0]
    results_key = routine[1]
    footer = routine[3]
    if partial:
        results_key += '_partial'

    results_key += f'_{append_to_results_key}'
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

                names = (syspar, savefolder)
                if reverse_order:
                    names = names[::-1]
                savename_ = '{}_{}_{}'.format(savename, *names)
                if merge:
                    data = np.zeros((len(vals), arr_shape), dtype=np.float64)

                    for i, value in enumerate(vals):

                        data[i] = get_fun(value[1], results_key,
                                          disorder_key, *args, **kwargs)

                    # sort according to disorder
                    data = data[data[:, 0].argsort()]
                    print(_join(savefolder_, savename_))
                    try:

                        np.savetxt(_join(savefolder_, savename_),
                                   data, footer=footer)
                    except ValueError:
                        print('Encountered error, file will not be saved!')
                        pass
                else:

                    data = []

                    for i, value in enumerate(vals):

                        result = get_fun(value[1], results_key,
                                         disorder_key, *args, **kwargs)

                        savename_temp = savename_ + '_{}_{}'.format(
                            disorder_key, value[0])
                        print(_join(savefolder_, savename_))
                        try:
                            np.savetxt(_join(savefolder_, savename_temp),
                                       result, footer=footer)
                        except ValueError:
                            print('Encountered error, file will not be saved!')
                            pass
