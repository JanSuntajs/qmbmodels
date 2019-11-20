"""
This module contains routines
for saving the eigenvalues files
to the disk.
"""

import numpy as np
import os
import json


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
