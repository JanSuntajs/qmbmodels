#!/usr/bin/env python


from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from _anderson_ententro import main_fun_entro
from qmbmodels.utils.cmd_parser_tools import arg_parser_general

_eentro_parse_dict = {'eentro_nstates': [int, -1],
                      'eentro_filling': [float, 0.5],
                      'eentro_partition': [float, 0.5]}

save_metadata = True

if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    # Obtain the command-line arguments relevant
    # to the entanglement entropy calculation:

    eentroDict, eentro_extra = arg_parser_general(_eentro_parse_dict)
    # number of many-body states for which we calculate
    # the entropy
    eentro_nstates = eentroDict['eentro_nstates']
    # filling fraction - ratio of occupied states
    # compared to the volume of the system
    filling = eentroDict['eentro_filling']

    partition_fraction = eentroDict['eentro_partition']

    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        print(argsDict)

        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization and entanglement calculation')

        eigvals, eentro = main_fun_entro(model, eentro_nstates, partition_fraction,
                                         filling)

        print('Displaying eigvals')
        print(eigvals)

        print('Displaying entanglement entropies')

        print(eentro)

        # ----------------------------------------------------------------------
        # save the files

        entanglement_name = (f'Entropy_noninteracting_'
                             f'nstates_{eentro_nstates}_'
                             f'partition_size_{partition_fraction:.2f}_'
                             f'filling_{filling:.2f}')
        # do not save field configurations
        eigvals_dict = {'Eigenvalues': eigvals,
                        entanglement_name: eentro}

        argsDict.update(eentroDict)
        syspar_keys += list(eentroDict.keys())
        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
