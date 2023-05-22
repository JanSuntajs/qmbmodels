#!/usr/bin/env python

"""
A module for performing full diagonalization
to calculate both the eigenvalues, the
ipr and the Renyi entanglement entropy for p last
spins.

We calculate the ipr and entanglement entropy for a range
of q values:

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
     2., 2.4, 2.6, 2.8., 3.];

Evidently, q=1 is missing above, since we need to use different
definitions both for IPR and the entanglement entropy. For the
former, we calculate the corresponding participation entropy
as the Shannon entropy of the eigenstate probability distribution.
Note that calculating the IPR for q=1 makes no sense as it would
trivially yield normalization. For the entanglement entropy,
we calculate the standard von Neumann entanglement entropy, which
is the limiting case of the Renyi entanglement entropy as q goes
to 1.

we concentrate on a selected number of states from the middle
of the spectrum, nstates = 1000.

We also calculate the entanglement measure based on the entanglement
entropy - the Renyi entropy. In this case, we do not store the
eigenvalues of the reduced density matrix, but we rather perform
calculations (sums of the eigenvalues raised to the appropriate
power) and then discard the eigenvalues.


# ---------------------
#
# CALLING THE SCRIPT
#
# ---------------------

Command-line argument for calling this script following
the corresponding submission script:

diag_ipr

Hence calling the program within qmbmodels would be:

python <submission_script.py> diag_ipr


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
qlist = np.append(np.arange(0.1, 2.1, 0.1), np.arange(2.2, 3.2, 0.2))
qlist = np.delete(qlist, 9)
# qlist_entro = np.append(np.arange(0.1, 1., 0.1), 2)
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

        # participation entropy for q = 1. -> Shannon entropy
        ipr_dict['ENTRO_PART_q_1.00'] = np.nansum(
            np.abs(eigvecs)**2 * np.log(np.abs(eigvecs)**2),
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

            for q in qlist:
                eentro_dict[f'EENTRO_RENYI_p_{p_:d}_q_{q:.2f}'] = \
                    (1./(1.-q)) * np.log(np.sum(
                        svd_vals**q, axis=1))
            # von neumann entropy for q = 1.
            eentro_dict[f'EENTRO_VN_p_{p_:d}'] = - \
                np.nansum(svd_vals * np.log(svd_vals), axis=1)

            schmidt_gap = np.abs(np.diff(svd_vals, axis=1))
            eentro_dict[f'SCHMIDT_GAP_p_{p_:d}'] = np.max(schmidt_gap, axis=1)

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
