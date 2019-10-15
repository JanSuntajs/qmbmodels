#!/usr/bin/env python

import os
import numpy as np
from ham1d.models.spin1d_kron import hamiltonian as hm

from utils import set_mkl_lib
from utils.cmd_parser_tools import arg_parser
# from . import utils as ut

if __name__ == '__main__':

    # first, read commandline args

    syspar_keys = ['L']
    modpar_keys = ['J', 'dJ', 'W', 'dW', 'Gamma', 'dGamma', 'seed']

    # argsDict -> system and module dependent parameters
    # extra -> path for saving the results
    argsDict, extra = arg_parser(syspar_keys, modpar_keys)
    print(extra)

    L = argsDict['L']
    J, dJ, W, dW, Gamma, dGamma, seed = [argsDict[key] for key in modpar_keys]
    print(argsDict)
    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']
    # syspar = ut.format_syspar_str(L)
    # modpar = ut.format_modpar_str(J, dJ, W, dW, Gamma, dGamma, seed)

    # --------------------------------------------------------------------
    print('Using seed: {}'.format(seed))
    np.random.seed(seed=seed)
    # set up the disordered terms
    W_dis = np.random.uniform(W - dW, W + dW, size=L)
    print(W_dis)
    Gamma_dis = np.random.uniform(Gamma - dGamma, Gamma + dGamma, size=L)
    print(Gamma_dis)
    J_dis = np.random.uniform(J - dJ, J + dJ, size=L)
    print(J_dis)

    # --------------------------------------------------------------------
    # S^z term -> a nested list of interaction parameters and
    # sites on which the operators act on
    h_z = [[W_dis[i], i] for i in range(L)]
    # S^x term
    gamma_x = [[Gamma_dis[i], i] for i in range(L)]
    # The two-site interaction term -> consider PBC
    J_zz = [[J_dis[i], i, (i + 1) % L] for i in range(L)]

    # ---------------------------------------------------------------------
    # define the interaction type for the corresponding
    # coupling lists
    ham_static = [['z', h_z], ['x', gamma_x], ['zz', J_zz]]

    # build the imbrie model hamiltonian
    imbrie_model = hm(L, ham_static, [])

    print('Starting diagonalization ...')
    # eigvals = imbrie_model.eigvals(turbo=True)
    eigvals, eigvecs = imbrie_model.eigsystem(turbo=True)
    print('Diagonalization finished!')

    print('Displaying eigvals')
    print(eigvals)

    # ----------------------------------------------------------------------
    # save the files

    # folder_path =
    filename = 'eigvals_{}_{}_seed_{}'.format(syspar, modpar, seed)
    print(filename)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    np.save(os.path.join(savepath, filename), eigvals)
