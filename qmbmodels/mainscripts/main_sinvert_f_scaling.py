#!/usr/bin/env python

"""
This code implements the PETSc and SLEPc
libraries in order to partially diagonalize
the quantum model hamiltonian of choice. For now,
the code is almost the same as the one in
main_sinvert.py except for calling the sinvert
function (at the very end), in which we have:

    sinvert_body(mod, argsDict, syspar, syspar_keys, modpar,
                 modpar_keys, mpirank, mpisize, comm, save_metadata,
                 savepath,
                 PETSc, SLEPc,
                 bipartition=bipartition, entropy_f_scaling=True)

so we have entropy_f_scaling = True instead of False such that the scaling
analysis of the entropy is performed. Rather than losing time
trying to make everything as terse and compact as possible,
we c/p-ed the existing code as this program will not be used
much anyway, only for some schematic plots. 


"""

from qmbmodels.utils.set_petsc_slepc import *
from qmbmodels.utils import set_mkl_lib

from qmbmodels.models.prepare_model import get_module_info

from _sinvert import sinvert_body

store_eigvecs = False
save_metadata = True
if __name__ == '__main__':

    comm = PETSc.COMM_WORLD

    mpisize = comm.size
    mpirank = comm.rank

    # extract all the relevant data from the command-line arguments
    # and prepare the model-specific parameters
    (mod, model_name, argsDict, seedDict, syspar_keys, modpar_keys,
     savepath, syspar, modpar, *rest) = get_module_info()

    
    print(argsDict)
    print('Using seed: {}'.format(argsDict['seed']))

    if model_name == 'rnd_grain':
        bipartition = 'last_four'
        print('Bipartitioning mode: last_four')
    else:
        bipartition = 'default'
        print('Bipartition mode: default.')
    sinvert_body(mod, argsDict, syspar, syspar_keys, modpar,
                 modpar_keys, mpirank, mpisize, comm, save_metadata,
                 savepath,
                 PETSc, SLEPc,
                 bipartition=bipartition, entropy_f_scaling=True)


