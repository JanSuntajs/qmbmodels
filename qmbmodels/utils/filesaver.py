"""
This module contains routines
for saving the eigenvalues files
to the disk as well as routines
for saving results of data analysis
routines such as r calculation.
"""

import numpy as np
import os
import json
import math


metalist = ['metadata', 'modpar', 'syspar']


def savefile(files, savepath, syspar, modpar, argsDict,
             syspar_keys, modpar_keys,
             diag_type='full',
             save_metadata=True,
             name='eigvals',
             save_type='npy'):
    """
    Prepare a file for saving.

    Parameters
    ----------

    files: dict or ndarray
                Can be either:
                a) A dictionary of arrays to be saved
                via the np.savez(). Intended to be used
                together with save_type = 'npz'
                b) A ndarray to be saved via the np.save
                method. Hence intended to use in the case
                when save_type = 'npy'.

    savepath: str
                Path to the folder where the eigenvalues
                are stored.

    syspar: str
                A string containing system parameters.

    modpar: str
                A string containing model parameters.

    syspar_keys: list
                A list containing system parameter keys.

    modpar_keys: list
                A list containing model parameter keys.

    diag_type: str, optional
                String specifying whether the spectrum
                was obtained using full or partial
                diagonalization. Hence, the options are
                'full' and 'partial'

    save_metadata: bool, optional
                Whether to store metadata or not.

    save_type: string
                Whether to save in the '.npy' or '.npz'
                file format. The two current options
                are thus:
                'npy'
                'npz'

    """
    if not os.path.isdir(savepath):

        os.mkdir(savepath)

    if diag_type == 'full':

        name = name

    elif diag_type == 'partial':

        name = 'partial_{}'.format(name)

    filename = '{}_{}_{}_seed_{}'.format(
        name, syspar, modpar, argsDict['seed'])

    print(filename)

    filepath = os.path.join(savepath, filename)
    if save_type == 'npy':
        np.save(filepath, files)
    elif save_type == 'npz':
        np.savez(filepath, **files)

    # prepare the metadata files
    if save_metadata:

        metapath = os.path.join(savepath, 'metadata')

        metafiles = [os.path.join(metapath,
                                  '{}_{}_{}.json'.format(name_,
                                                         syspar, modpar))
                     for name_ in metalist]

        argsDict_ = argsDict.copy()
        argsDict_.pop('seed', None)

        sys_dict = {key: value for key, value in argsDict_.items() if
                    key in syspar_keys}
        mod_dict = {key: value for key, value in argsDict_.items() if
                    key in modpar_keys}

        argsDict_ = {key: value for key, value in argsDict_.items() if
                     key not in syspar_keys + modpar_keys}

        dicts = [argsDict_, sys_dict, mod_dict]

        for file, dict_ in zip(metafiles, dicts):
            if not os.path.isfile(file):
                with open(file, "w") as f:

                    jsonfile = json.dumps(dict_)
                    f.write(jsonfile)


def save_hdf_datasets(datasetdict, file, attrs):
    """
    A routine for creating new
    hdf5 datasets if those are not
    already present in the hdf5 file,
    or appending/replacing data in
    the existing datasets.

    Parameters:
    -----------

    datasetdict: dictionary
        A dictionary of pairs:
        namestring: [data, maxshape]
        Where:
        namestring is a string - the name of the dataset
        to be saved or appended to

        data is a (typically 2D) ndarray of the actual
        numerical data.
        maxshape is typically a 2-tuple indicating the
        maximum dimensions of the data array.
    """

    for key, value in datasetdict.items():

        data = value[0]
        maxshape = value[1]

        if key not in file.keys():

            file.create_dataset(key, data=data,
                                maxshape=maxshape)

        else:

            file[key][()] = data

        for key1, value1 in attrs.items():

            file[key].attrs[key1] = value1


def save_external_files(filename, savedict):
    """
    A routine for saving external txt files with results
    for faster access and reading without the need for
    opening the hdf5 files.

    Parameters:
    -----------

    filename: string
            Filename of the hdf5 file storing the data.

    savedict: dictionary
              A dictonary with keys specifying the
              filenames of the external files to be
              saved and the corresponding values refer
              to the numerical data to be saved.

    NOTE: the external file's filename matches the name
    of the original .hdf5 file, except that the 'eigvals'
    prefix has been replaced by the selected key from
    the savedict dictionary.
    """

    for key, value in savedict.items():
        head, tail = os.path.split(filename)
        txt_file = tail.replace('eigvals', key)
        txt_file = txt_file.replace('.hdf5', '.txt')
        print(txt_file)
        print(f'{head}/{txt_file}')
        np.savetxt(f'{head}/{txt_file}', value)


def load_eigvals(file, setnames, partial=True, nener=-1):
    """
    A function for loading the dataset storing
    the energy eigenvalues and storing them
    into memory for further analysis. Also
    enables slicing of the array in case only
    particular samples or portions of the spectra
    are to be analysed.

    Parameters:
    -----------

    file: hdf5 file
          hdf5 file containing the 'Eigvals' or
          'Eigvals_partial' dataset which is to be loaded.

    setnames: list or 1D array-like object
            A list of dataset names for datasets which are
            to be obtained by performing some kind of analysis
            on the eigenvalues. In cases where only a portion
            of the eigenvalues from the spectra is chosen
            (when nener is not None), a suffix '_nev_{number_of_
            chosen_energies}' is appended to all the setnames
            to indicate that the quantities of interest have
            been calculated on a portion of the spectrum.

    partial: boolean, optional
            Indicates whether the partial (if partial==True)
            or full diagonalization results (if partial==False)
            are under consideration. Defaults to True.

    nener: int
            If only a portion of the eigenstates is to be considered,
            this should be an integer specifying the number of eigenvalues
            to be selected. We select the values from the center of
            the spectra and discard the remainder as symmetrically as
            possible. In case nener is greater than the actual Hilbert
            space dimension, the value of nener is reset to the number
            of available states. In case nener is -1, all states are
            selected.

    Returns:

    data: ndarray, 2D
            ndarray with the eigenvalues for different spectra.
    attrs: dict,
            attributes of the eigenvalues dataset.
    setnames: list
              See above of explanation. In case nener is not None, an
              additional suffix is appended, as explained above.
    """

    if type(nener) is not int:
        raise TypeError(('load_eigvals error: '
                         'nener type should be int!'))
    if nener < -1:
        raise ValueError(('load_eigvals error: '
                          'nener should be greater than -1!'))
    eigname = 'Eigenvalues'
    if partial:
        eigname += '_partial'

    # load the eigenvalues into memory
    data = file[eigname][:]
    # slice if needed
    attrs = dict(file[eigname].attrs)

    # number of
    nener_orig = attrs['nener']
    if nener is not -1:
        if nener > nener_orig:
            nener = nener_orig
        remain = nener_orig - nener
        start = math.ceil(remain / 2.)
        stop = start + nener
        data = data[:, slice(start, stop)]

        attrs['nener'] = nener

        setnames = [setname + f'_nev_{int(nener)}' for
                    setname in setnames]
    return data, attrs, setnames
