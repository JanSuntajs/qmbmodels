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


def prepare_files(filenames, partial=False):
    """ 
    Function that takes care of the fact
    that partial diagonalization does
    not always return the same number
    of eigenvalues -> hence it reshapes
    the whole list of values so that shapes
    of individual 1D spectra match.
    """

    files = [np.load(file) for file in filenames]

    if partial:

        files = [file for file in files if file.size != 0]

        shapes = [file.shape[0] for file in files]
        minshape = np.min(shapes)
        files = [file[:minshape] for file in files]

    return np.array(files)


if __name__ == '__main__':

    partial = False
    eigenkey = 'Eigenvalues'
    fnamekey = 'Eigenvalues_filenames'

    print(os.getcwd())
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    # load metadata

    metapath = os.path.join(savepath, 'metadata')
    metafile, sysfile, modfile = [glob.glob(f"{metapath}/{name}*.json")[0]
                                  for name in metalist]

    files = [metafile, sysfile, modfile]

    for i, file in enumerate(files):
        with open(file) as json_file:

            dicts[i].update(json.load(json_file).copy())
            print(dicts[i])

    # Load the eigenvalue files, then save them to hdf5
    loadpath = savepath

    filenames_ = glob.glob(f"{loadpath}/*.npy")

    # files = np.array([np.load(file) for file in filenames_])

    filenames = np.array([os.path.split(name)[1]
                          for name in filenames_], dtype=object)

    filename = filenames[0].split('_seed', 1)[0]

    # in case we are dealing with the partial diagonalization case,
    # change the partial flag accordingly
    if 'partial' in filename:

        partial = True
        eigenkey = 'Eigenvalues_partial'
        fnamekey = 'Eigenvalues_partial_filenames'

    files = prepare_files(filenames_, partial)

    filename = filename.replace('partial_', '')

    nsamples, nener = files.shape

    datasets = {eigenkey: files,
                fnamekey: filenames}

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
    #  ----------------------------------------------------------------------
    # save the files to hdf5

    # if not os.path.isdir(savepath):
    #     os.makedirs(savepath)

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
                if key == eigenkey:
                    if not partial:
                        maxshape = (None, nener)
                    else:
                        maxshape = (None, None)

                    eigset = f.create_dataset(
                        key, data=value, maxshape=maxshape)

                # this is aY¸ numpy array of the object datatype
                else:
                    string_dt = h5py.special_dtype(vlen=str)

                    f.create_dataset(
                        key, data=value, maxshape=maxshape, dtype=string_dt)

            # append attributes
            for key, value in attrs.items():

                f[eigenkey].attrs[key] = value

            print('created actual values')

        # append to the existing datasets if the datasets already exist
        else:
            print('Appending spectra to the existing values!')
            nsamples0 = f[eigenkey].shape[0]

            # check for duplicates
            indices = np.array([i for i, name in enumerate(filenames)
                                if name not in
                                f[fnamekey][()]], dtype=np.int8)

            datasets[eigenkey] = files[indices, :]
            datasets[fnamekey] = filenames[indices]
            nsamples = indices.shape[0]

            nsamples += nsamples0
            attrs['nsamples'] = nsamples

            if partial:
                nener_ = f[eigenkey].shape[1]
                if nener > nener_:
                    nener = nener_
            f[eigenkey].resize((nsamples, nener))

            f[eigenkey][nsamples0:, :] = datasets[eigenkey]
            f[fnamekey].resize((nsamples,))
            f[fnamekey][nsamples0:] =  \
                datasets[fnamekey]

            # if attributes have also changed
            for key, value in attrs.items():

                f[eigenkey].attrs[key] = value

    for filename in filenames_:

        os.remove(filename)
