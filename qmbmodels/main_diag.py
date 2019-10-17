#!/usr/bin/env python

import os
import numpy as np
import json

from ham1d.models.spin1d_kron import hamiltonian as hm

from utils import set_mkl_lib
from utils.cmd_parser_tools import arg_parser
from models import heisenberg

if __name__ == '__main__':

    syspar_keys = heisenberg.syspar_keys
    modpar_keys = heisenberg._modpar_keys

    # argsDict -> system and module dependent parameters
    # extra -> path for saving the results
    argsDict, extra = arg_parser(syspar_keys, modpar_keys)

    # define attributes for the hdf5

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    print('Using seed: {}'.format(argsDict['seed']))

    # get the instance of the appropriate hamiltonian
    # class and the diagonal random fields used
    model, fields = heisenberg.construct_hamiltonian(argsDict)

    print('Starting diagonalization ...')
    # eigvals = imbrie_model.eigvals(turbo=True)
    eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
    print('Diagonalization finished!')

    print('Displaying eigvals')
    print(eigvals)

    # ----------------------------------------------------------------------
    # save the files
    filename = 'eigvals_{}_{}_seed_{}'.format(syspar, modpar,
                                              argsDict['seed'])
    print(filename)
    # if not os.path.isdir(savepath):
    #     os.makedirs(savepath)

    np.save(os.path.join(savepath, filename), eigvals)

    # prepare the metadata_files
    metapath = os.path.join(savepath, 'metadata')

    metafiles = [os.path.join(metapath,
                              '{}_{}_{}.json'.format(name, syspar, modpar))
                 for name in ['metadata', 'modpar', 'syspar']]

    # if not os.path.isdir(metapath):

    argsDict_ = argsDict.copy()
    argsDict_.pop('seed', None)

    sys_dict = {key: value for key, value in argsDict_.items() if
                key in syspar_keys}
    mod_dict = {key: value for key, value in argsDict_.items() if
                key in modpar_keys}

    argsDict_ = {key: value for key, value in argsDict_.items() if
                 key not in syspar_keys + modpar_keys}

    dicts = [argsDict_, sys_dict, mod_dict]

    #     os.makedirs(metapath, exist_ok=True)
    for file, dict_ in zip(metafiles, dicts):
        if not os.path.isfile(file):
            with open(file, "w") as f:

                jsonfile = json.dumps(dict_)
                f.write(jsonfile)

        # folder_path =
