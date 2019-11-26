#!/usr/bin/env python

import os
import numpy as np
import json

#from ham1d.models.spin1d_kron import hamiltonian as hm

from utils import set_mkl_lib
from utils.cmd_parser_tools import arg_parser
from utils.filesaver import savefile
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
    eigvals = model.eigvals(complex=False)
    # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
    print('Diagonalization finished!')

    print('Displaying eigvals')
    print(eigvals)

    # ----------------------------------------------------------------------
    # save the files
    eigvals_dict = {'Eigenvalues': eigvals,
                    'Hamiltonian_random_disorder': fields}
    savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
             syspar_keys, modpar_keys, 'full', True, save_type='npz')
