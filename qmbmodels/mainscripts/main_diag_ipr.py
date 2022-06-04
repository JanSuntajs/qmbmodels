#!/usr/bin/env python

"""
A module for performing full diagonalization
to calculate both the eigenvalues and
the ipr. We calculate the ipr for a range
of q values:

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2];

we concentrate on a selected number of states from the middle
of the spectrum, nstates = 1000.

We also calculate the entanglement measure based on the entanglement
entropy.


"""
from re import I
import numpy as np

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from ham1d.entropy.ententro import Entangled

save_metadata = True

nener = 1000

# q values for the ipr calculation
qlist = np.append(np.arange(0.1, 1., 0.1), 2)
qlist_entro = np.append(np.arange(0.1, 1., 0.1), 2)
# partitions -> how many of the farthermost spins
# to include
plist = [1, 2, 3, 4]


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
        eigvals, eigvecs = model.eigsystem(complex=False)

        # slice the eigvecs array so that only
        # 1000 states from the middle of the spectra
        # are taken
        nstates = model.nstates
        if nener > nstates:
            nener = nstates

        # select only a portion of eigstates
        eigvecs = eigvecs[:, int(0.5 * (nstates - nener))                          : int(0.5*(nstates + nener))]

        # --------------------------------------------------
        #
        #  IPR calculation
        #
        # ---------------------------------------------------
        ipr_dict = {}
        for q in qlist:
            ipr_dict[f'IPR_q_{q:.2f}'] = np.sum(
                np.abs(eigvecs)**(2*q),
                axis=0)

        # ---------------------------------------------------
        #
        #   Ententro-based calculation
        #
        # ---------------------------------------------------
        eentro_dict = {}
        for p_ in plist:

            # eigvals of the svd decomposition for each
            # eigstate
            svd_vals = []
            for eigvec in eigvecs.T:
                entangled = Entangled(eigvec, argsDict['L'], p_)
                entangled.partitioning('homogenous')
                entangled.svd()
                # _s_coeffs are the coefficients of the
                # sdv decomposition
                svd_vals.append(entangled._s_coeffs)
            svd_vals = np.array(svd_vals)**2

            for q in qlist_entro:
                eentro_dict[f'EENTRO_RENYI_p_{p_:d}_q_{q:.2f}'] = (1./(1.-q)) * np.log(np.sum(
                    svd_vals**q, axis=1))
            
            eentro_dict[f'EENTRO_VN_p_{p_:d}'] = - np.nansum( svd_vals * np.log(svd_vals))

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
                        **eentro_dict,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
