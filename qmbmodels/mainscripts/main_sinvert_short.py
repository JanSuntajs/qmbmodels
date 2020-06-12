#!/usr/bin/env python

"""
This code implements the PETSc and SLEPc
libraries in order to partially diagonalize
the quantum model hamiltonian of choice. While
the general structure of the code allows for choice
of different spectral transformations and different
solver contexts, the main intent of this code is to
obtain the eigenvalues and the eigenvectors from
a selected region of the hamiltonian's spectrum
and to also calculate the eigenvector's entanglement
entropy.

The suffix short indicates the job is intended for
running shorter jobs using a for loop since sending
a large number of short jobs to a SLURM cluster can
cause a significant overlay and slowning down of the
scheduling system.
"""

from qmmodels.utils.set_petsc_slepc import *
from qmbmodels.utils import set_mkl_lib

from qmbmodels.models.prepare_model import get_module_info

from ._sinvert import sinvert_body

store_eigvecs = False

if __name__ == '__main__':

    comm = PETSc.COMM_WORLD
    mpirank = 0
    mpisize = 1

    # extract all the relevant data from the command-line arguments
    # and prepare the model-specific parameters
    (mod, model_name, argsDict, seedDict, syspar_keys, modpar_keys,
        savepath, syspar, modpar, min_seed, max_seed) = get_module_info()

    save_metadata = True
    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed

        sinvert_body(mod, argsDict, syspar, syspar_keys, modpar,
                     modpar_keys, mpirank, mpisize, comm, save_metadata,
                     savepath,
                     PETSc, SLEPc)

        save_metadata = False
