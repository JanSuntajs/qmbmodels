#!/usr/bin/env python


from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from _anderson_ententro import main_fun_entro

save_metadata = True

if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed

        eentro_nstates = argsDict['eentro_nstates']
        filling = argsDict['filling_fraction']
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization and entanglement calculation')

        eigvals, eentro = main_fun_entro(model, eentro_nstates, filling)

        print('Displaying eigvals')
        print(eigvals)

        print('Displaying entanglement entropies')

        print(eentro)

        # ----------------------------------------------------------------------
        # save the files

        entanglement_name = (f'Entanglement_entropy_'
                             f'Anderson_filling_fraction_{filling:.2f}')
        # do not save field configurations
        eigvals_dict = {'Eigenvalues': eigvals,
                        entanglement_name: eentro}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
