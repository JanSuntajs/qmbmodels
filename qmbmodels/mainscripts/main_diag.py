#!/usr/bin/env python


from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

save_metadata = True

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
        eigvals = model.eigvals(complex=False)

        parity = model.parity_shuffle()
        print(parity)
        # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
        print('Diagonalization finished!')

        print('Displaying eigvals')
        print(eigvals)

        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals,
                        **fields}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
