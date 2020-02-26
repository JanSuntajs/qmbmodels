#!/usr/bin/env python
"""
This module implements a routine for saving files in a hdf5 file
after the batch calculation routines have finished.


"""

import os
import sys
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
    A function that takes care of the
    fact that partial diagonalization
    does not always return the same
    number of eigenvalues and that
    individual datasets also do not
    necessairily have the same shapes
    (the data about the random disorder
    fields and the eigenspectra results
    differ considerably in that respect,
    for instance). Hence this function
    takes care of properly reshaping the
    data which is particularly important
    once the new data are appended to
    the exsisting ones.

    Parameters:
    -----------

    filenames: list, str
               A list of full paths to the results files


    """

    files = [np.load(file) for file in filenames]

    # the loaded files are stored in numpy's .npz
    # format. We first extract the keys describing
    # the files' contents
    filekeys = files[0].files

    # get the metadata -> if 'Metadata' string is
    # present a key from the filekeys list
    metakeys = [key for key in filekeys if 'Metadata' in key]

    # everything else are results
    reskeys = [key for key in files[0].files if 'Metadata' not in key]
    print('reskeys')
    print(reskeys)
    results_dict = {}
    # shapes of the arrays within the results_dict
    shapes_dict = {}

    filenames, filename, partial = format_filenames(filenames)

    # make sure that the keys are renamed if needed -> in the
    # case of partial diagonalization results
    append_part = ''
    if partial:
        append_part = '_partial'

    results_dict['Eigenvalues_filenames'] = filenames
    shapes_dict['Eigenvalues_filenames'] = filenames.shape

    # metadata dictionary
    for key in metakeys:

        results_dict[key] = np.array([file[key] for file in files])
        shapes_dict[key] = results_dict[key].shape

    if partial:
        for key in reskeys:

            files_ = [file[key] for file in files if file[key].size != 0]

            shapes = [file.shape[0] for file in files_]
            minshape = np.min(shapes)
            files_ = [file[:minshape] for file in files_]
            results_dict[key] = np.array(files_)
            shapes_dict[key] = results_dict[key].shape
    # just fill the dictionary if all the entries are supposed to be
    # of the same shape
    else:
        for key in reskeys:

            results_dict[key] = np.array([file[key] for file in files])
            shapes_dict[key] = results_dict[key].shape

    # add '_partial' suffix for partial diagonalization results
    for key in results_dict.keys():

        if (partial and 'partial' not in key):

            for dict_ in [results_dict, shapes_dict]:
                newkey = key + append_part
                dict_[newkey] = dict_.pop(key)

    key_specifiers = {}

    key_specifiers['fnamekeys'] = [
        key for key in results_dict.keys() if 'filenames' in key]
    # key_specifiers['fields'] = [key for key in results_dict.keys() if
    #                             'random_disorder' in key]
    key_specifiers['metakeys'] = [
        key for key in results_dict.keys() if 'Metadata' in key]

    exclude = ['filenames', 'Metadata']
    key_specifiers['reskeys'] = [key for key in results_dict.keys()
                                 if all([(exc not in key) for exc in exclude])]

    print('key_specifiers')
    print(key_specifiers)
    return results_dict, shapes_dict, key_specifiers


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

    try:
        metafile, sysfile, modfile = [glob.glob(f"{metapath}/{name}*.json")[0]
                                      for name in metalist]
    except IndexError:
        print('Metadata files not present! Exiting')
        sys.exit(0)

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

    if filenames_:

        filenames, filename, partial = format_filenames(filenames_)

        # eigenkey, fnamekey = full_partial(partial)
        datasets, shapes_dict, key_specifiers = prepare_files(filenames_)
        print(shapes_dict)
        print(key_specifiers)

        enerkey = [key for key in key_specifiers['reskeys']
                   if (key == 'Eigenvalues' or
                       key == 'Eigenvalues_partial')][0]
        # disorder_key = key_specifiers['fields'][0]
        fnamekey = key_specifiers['fnamekeys'][0]
        nsamples, nener = shapes_dict[enerkey]

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
                            maxshape = (None, shapes_dict[key][1])
                        else:
                            maxshape = (None, None)

                        eigset = f.create_dataset(
                            key, data=value, maxshape=maxshape)

                    # this is numpy array of the object datatype
                    elif key in key_specifiers['fnamekeys']:
                        string_dt = h5py.special_dtype(vlen=str)

                        f.create_dataset(
                            key, data=value, maxshape=maxshape,
                            dtype=string_dt)

                for reskey in key_specifiers['reskeys']:
                    for key, value in attrs.items():

                        f[reskey].attrs[key] = value

                print('created actual values')

            # append to the existing datasets if the datasets already exist
            else:
                print('Appending spectra to the existing values!')

                filenames_strip = [_strip(filename)
                                   for filename in f[fnamekey][()]]
                indices = np.array([i for i, name in enumerate(filenames)
                                    if _strip(name) not in
                                    filenames_strip], dtype=np.int8)
                datasets[fnamekey] = filenames[indices]

                for reskey in key_specifiers['reskeys']:
                    if reskey not in f.keys():
                        pass
                    else:
                        nsamples = indices.shape[0]
                        nsamples0 = f[reskey].shape[0]

                        datasets[reskey] = datasets[reskey][indices, :]

                        nsamples += nsamples0
                        attrs['nsamples'] = nsamples
                        orig_shape = f[reskey].shape[1]
                        shape_resize = orig_shape
                        if partial:
                            if shapes_dict[reskey][1] < orig_shape:
                                shape_resize = shapes_dict[reskey][1]
                            elif shapes_dict[reskey][1] > orig_shape:
                                datasets[reskey] = datasets[reskey][:,
                                                                    :orig_shape]
                        f[reskey].resize((nsamples, shape_resize))

                        f[reskey][nsamples0:, :] = datasets[reskey]

                        # if attributes have also changed
                        for key, value in attrs.items():

                            f[reskey].attrs[key] = value

                f[fnamekey].resize((nsamples,))

                f[fnamekey][nsamples0:] =  \
                    datasets[fnamekey]
        # -------------------------------------------------
        #
        #
        #           REMOVAL OF THE ORIG. NPZ OR NPY FILES
        #
        #
        # -------------------------------------------------
        for filename in filenames_:

            os.remove(filename)

    else:
        print('No .npz files present in the folder!')

    if ((not glob.glob(f'{loadpath}/*.hdf5')) and
            (not glob.glob(f'{loadpath}/*.npz'))):
        print(f'Removing the folder {loadpath}!')
        os.rmdir(f'{loadpath}')
