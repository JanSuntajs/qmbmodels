#!/usr/bin/env python

"""
A module for performing full diagonalization
to calculate both the eigenvalues and
the ipr. We calculate the ipr for a range
of q values:

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2];

we concentrate on a selected number of states from the middle
of the spectrum, nstates = 1000


"""
from re import I
import numpy as np

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

save_metadata = True

nener = 1000

if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization ...')
        eigvals, eigvecs = model.eigsystem(complex=True)

        # slice the eigvecs array so that only
        # 1000 states from the middle of the spectra
        # are taken
        nstates = model.nstates
        if nener > nstates:
            nener = nstates

        ipr_dict = {}
        qlist = np.append(np.arange(0.1, 1., 0.1), 2)
        for q in qlist:
            ipr_dict[f'IPR_q_{q:.2f}'] = np.sum(
                np.abs(eigvecs[:, int(0.5*(nstates - nener)):
                               int(0.5*(nstates + nener))])**(2*q),
                axis=0)

        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization finished!')

        print('Displaying eigvals')
        print(eigvals)

        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals,
                        'EIG_IPR': eigvals[int(0.5 * (nstates -
                                                    nener)):
                                           int(0.5 * (nstates + nener))],
                        **ipr_dict,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
