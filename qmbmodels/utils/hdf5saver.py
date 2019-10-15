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


if __name__ == '__main__':

    print(os.getcwd())
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    # load metadata

    metapath = os.path.join(savepath, 'metadata')
    metafile = glob.glob(f"{metapath}/*.json")[0]

    with open(metafile) as json_file:

        metadata = json.load(json_file).copy()

    # Load the eigenvalue files, then save them to hdf5
    loadpath = savepath

    filenames_ = glob.glob(f"{loadpath}/*.npy")

    files = np.array([np.load(file) for file in filenames_])

    filenames = np.array([os.path.split(name)[1]
                          for name in filenames_], dtype=object)
    filename = filenames[0].split('_seed', 1)[0]
    print(filenames)
    print(filename)
    nsamples, nener = files.shape

    datasets = {'Eigenvalues': files,
                'Eigenvalues_filenames': filenames}

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

    }

    attrs = {'nener': nener,
             'nsamples': nsamples,
             }

    system_info = {'syspar': syspar,
                   'modpar': modpar}

    misc = {**attrs, **system_info, **metadata, **creator}

    #  ----------------------------------------------------------------------
    # save the files to hdf5

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    filename = os.path.join(savepath, filename + '.hdf5')

    # save a hdf5 file
    with h5py.File(filename, 'a') as f:

        # if 'misc', 'metadata' and 'system_info' were
        # not yet created

        if all([(key not in f.keys()) for key in datasets.keys()]):
            print('Creating the dataset!')

            # now add the values for the first time
            for key, value in datasets.items():
                maxshape = (None,)
                # eigenvalues are stored as a numpy array
                if key == 'Eigenvalues':
                    maxshape = (None, nener)

                    eigset = f.create_dataset(
                        key, data=value, maxshape=maxshape)

                # this is a numpy array of the object datatype
                else:
                    string_dt = h5py.special_dtype(vlen=str)

                    f.create_dataset(
                        key, data=value, maxshape=maxshape, dtype=string_dt)

            for key, value in misc.items():

                f['Eigenvalues'].attrs[key] = value

            print('created actual values')
        # append to the existing datasets if the datasets already exist
        else:
            print('Appending spectra to the existing values!')
            nsamples0 = f['Eigenvalues'].shape[0]

            # check for duplicates
            indices = np.array([i for i, name in enumerate(filenames)
                                if name not in
                                f['Eigenvalues_filenames'][()]], dtype=np.int8)

            datasets['Eigenvalues'] = files[indices, :]
            datasets['Eigenvalues_filenames'] = filenames[indices]
            nsamples = indices.shape[0]

            nsamples += nsamples0
            misc['nsamples'] = nsamples

            f['Eigenvalues'].resize((nsamples, nener))
            f['Eigenvalues'][nsamples0:, :] = datasets['Eigenvalues']
            f['Eigenvalues_filenames'].resize((nsamples,))
            f['Eigenvalues_filenames'][nsamples0:] =  \
                datasets['Eigenvalues_filenames']

            for key, value in misc.items():

                f['Eigenvalues'].attrs[key] = value

    for filename in filenames_:

        os.remove(filename)
