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
    print(extra)

    print(argsDict)
    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    print('Using seed: {}'.format(argsDict['seed']))

    # get the instance of the appropriate hamiltonian
    # class and the diagonal random fields used
    model, fields = heisenberg.construct_hamiltonian(argsDict)

    print('Starting diagonalization ...')
    # eigvals = imbrie_model.eigvals(turbo=True)
    eigvals, eigvecs = model.eigsystem(turbo=True)
    print('Diagonalization finished!')

    print('Displaying eigvals')
    print(eigvals)

    # ----------------------------------------------------------------------
    # save the files
    filename = 'eigvals_{}_{}_seed_{}'.format(syspar, modpar,
                                              argsDict['seed'])
    print(filename)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    np.save(os.path.join(savepath, filename), eigvals)

    # prepare the metadata_files
    metapath = os.path.join(savepath, 'metadata')
    metafile = os.path.join(metapath,
                            'metadata_{}_{}.json'.format(syspar, modpar))

    if not os.path.isdir(metapath):

        os.makedirs(metapath, exist_ok=True)

    if not os.path.isfile(metafile):
        with open(metafile, "w") as f:
            argsDict_ = argsDict.copy()
            argsDict_.pop('seed', None)
            json = json.dumps(argsDict_)
            f.write(json)

        # folder_path =
