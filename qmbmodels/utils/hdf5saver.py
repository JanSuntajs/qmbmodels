#!/usr/bin/env python
"""
This module implements a routine for saving files in a hdf5 file
after the batch calculation routines have finished.


"""

import os
import numpy as np
import glob
import time
import json

# from dataIO import hdf5save
import h5py
from cmd_parser_tools import arg_parser
from filesaver import metalist

metadata = {}
sysdict = {}
moddict = {}

dicts = [metadata, sysdict, moddict]


def _strip(name):
    """
    Strip fileformat specifiers
    from a name.
    """

    return name.strip('.npy').strip('.npz')


def format_filenames(filenames):
    """
    A function that takes care of proper
    formatting of the filenames.

    Parameters:
    -----------

    filenames: list, str
               A list of strings specifying full paths
               to the files in question

    Returns:
    --------

    filenames_: ndarray, dtype=object
                A list of formatted filename strings,
                where only the path's tail is retained.
                object dtype is chosen for compatibility
                with hdf5 protocols
    filename: str
                The file descriptor string common to all
                files corresponding to the same system and
                model parameters.
    partial: boolean
                Determines whether we are dealing with
                the results of partial of full diagonalization.


    """
    filenames_ = np.array([os.path.split(name)[1]
                           for name in filenames], dtype=object)

    filename = filenames_[0].split('_seed', 1)[0]

    if 'partial' in filename:

        partial = True
    else:
        partial = False

    filename = filename.replace('partial_', '')
    return filenames_, filename, partial


def prepare_files(filenames):
    """
    Function that takes care of the fact
    that partial diagonalization does
    not always return the same number
    of eigenvalues -> hence it reshapes
    the whole list of values so that shapes
    of individual 1D spectra match.

    Returns a dictionary where arrays of results
    and additional information ('Metadata') correspond
    to their keys.

    Parameters:
    -----------

    filenames: list, str
               A list of full paths to the results files

    Returns:
    --------

    results_dict: dict
               A dict of keys providing human-readable
               descriptions, while the values are their
               corresponding data arrays.
    key_specifiers: dict
               A dict which specifies which keys in the
               results dict represent results ('reskeys'
               entry), which ones are for metadata
               ('metakeys') and which ones for filenames
               lists ('fnamekeys')
    """

    files = [np.load(file) for file in filenames]
    filekeys = files[0].files

    # keys corresponding to the metadata
    metakeys = [key for key in filekeys if 'Metadata' in key]
    # the numerical results
    reskeys = [key for key in files[0].files if 'Metadata' not in key]
    results_dict = {}

    filenames, filename, partial = format_filenames(filenames)

    # make sure that the keys are renamed if needed -> in the
    # case of partial diagonalization results
    append_part = ''
    if partial:
        append_part = '_partial'

    results_dict['Eigenvalues_filenames'] = filenames

    # metadata dictionary
    for key in metakeys:

        results_dict[key] = [file[key] for file in files]

    # reshape the files if partial diagonalization is in order
    if partial:
        for key in reskeys:

            files_ = [file[key] for file in files if file[key].size != 0]

            shapes = [file.shape[0] for file in files_]
            minshape = np.min(shapes)
            files_ = [file[:minshape] for file in files_]
            results_dict[key] = np.array(files_)
    # just fill the dictionary if all the entries are supposed to be
    # of the same shape
    else:
        for key in reskeys:

            results_dict[key] = files[key]

    # add '_partial' suffix for partial diagonalization results
    for key in results_dict.keys():

        if (partial and 'partial' not in key):

            results_dict[key + append_part] = results_dict.pop(key)

    key_specifiers = {}

    key_specifiers['fnamekeys'] = [
        key for key in results_dict.keys() if 'filenames' in key]
    key_specifiers['metakeys'] = [
        key for key in results_dict.keys() if 'Metadata' in key]
    key_specifiers['reskeys'] = [key for key in results_dict.keys()
                                 if (('filenames' not in key) and
                                     'Metadata' not in key)]

    return results_dict, key_specifiers


if __name__ == '__main__':

    print(os.getcwd())
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    # --------------------------------------------------------------------
    #
    #
    #                        METADATA LOADING
    #
    #
    # --------------------------------------------------------------------

    metapath = os.path.join(savepath, 'metadata')
    metafile, sysfile, modfile = [glob.glob(f"{metapath}/{name}*.json")[0]
                                  for name in metalist]

    files = [metafile, sysfile, modfile]

    for i, file in enumerate(files):
        with open(file) as json_file:

            dicts[i].update(json.load(json_file).copy())
            print(dicts[i])

    # --------------------------------------------------------------------
    #
    #
    #                        LOADING THE NUMERICAL DATA
    #
    #
    # --------------------------------------------------------------------
    loadpath = savepath

    filenames_ = glob.glob(f"{loadpath}/*.npz")

    filenames, filename, partial = format_filenames(filenames_)

    # eigenkey, fnamekey = full_partial(partial)
    datasets, key_specifiers = prepare_files(filenames_)
    print(key_specifiers)

    enerkey = [key for key in key_specifiers['reskeys']
               if (key == 'Eigenvalues' or key == 'Eigenvalues_partial')][0]
    fnamekey = key_specifiers['fnamekeys'][0]
    nsamples, nener = datasets[enerkey].shape

    # datasets = {eigenkey: files,
    #             fnamekey: filenames}

    # information about the creator of the data
    creator = {

        'OS': os.name,
        'User': 'Jan Suntajs',
        'email': 'Jan.Suntajs@ijs.si',
        'institute': 'IJS F1',
        'Date': time.time(),
        'dict_format': 'json',
        'loadpath': loadpath,
        'filename': filename,
        **metadata

    }

    attrs = {'nener': nener,
             'nsamples': nsamples,
             'syspar': syspar,
             'modpar': modpar,
             **sysdict,
             **moddict
             }
    # --------------------------------------------------------------------
    #
    #
    #                        HDF5 SAVING
    #
    #
    # --------------------------------------------------------------------

    filename = os.path.join(savepath, filename + '.hdf5')

    # save a hdf5 file
    with h5py.File(filename, 'a') as f:

        # attributes of the whole file
        for key, value in creator.items():

            f.attrs[key] = value

        # if 'misc', 'metadata' and 'system_info' were
        # not yet created -> the file was not opened before
        if all([(key not in f.keys()) for key in datasets.keys()]):
            print('Creating the dataset!')

            # now add the values for the first time
            for key, value in datasets.items():
                maxshape = (None,)
                # eigenvalues are stored as a numpy array
                if key in key_specifiers['reskeys']:
                    if not partial:
                        maxshape = (None, nener)
                    else:
                        maxshape = (None, None)

                    eigset = f.create_dataset(
                        key, data=value, maxshape=maxshape)

                # this is aY¸ numpy array of the object datatype
                elif key in key_specifiers['fnamekeys']:
                    string_dt = h5py.special_dtype(vlen=str)

                    f.create_dataset(
                        key, data=value, maxshape=maxshape, dtype=string_dt)

            for reskey in key_specifiers['reskeys']:
                for key, value in attrs.items():

                    f[reskey].attrs[key] = value

            print('created actual values')

        # append to the existing datasets if the datasets already exist
        else:
            print('Appending spectra to the existing values!')
            for reskey in key_specifiers['reskeys']:
                nsamples0 = f[reskey].shape[0]

                # check for duplicates
                filenames_strip = [_strip(filename)
                                   for filename in f[fnamekey][()]]
                indices = np.array([i for i, name in enumerate(filenames)
                                    if _strip(name) not in
                                    filenames_strip], dtype=np.int8)

                datasets[reskey] = datasets[reskey][indices, :]
                datasets[fnamekey] = filenames[indices]
                nsamples = indices.shape[0]

                nsamples += nsamples0
                attrs['nsamples'] = nsamples

                if partial:
                    nener_ = f[reskey].shape[1]
                    if nener > nener_:
                        nener = nener_
                f[reskey].resize((nsamples, nener))

                f[reskey][nsamples0:, :] = datasets[reskey]
                f[fnamekey].resize((nsamples,))
                f[fnamekey][nsamples0:] =  \
                    datasets[fnamekey]

                # if attributes have also changed
                for key, value in attrs.items():

                    f[reskey].attrs[key] = value

    # -------------------------------------------------
    #
    #
    #           REMOVAL OF THE ORIG. NPZ OR NPY FILES
    #
    #
    # -------------------------------------------------
    for filename in filenames_:

        os.remove(filename)
