#!/usr/bin/env python

"""
Module with functions for calculations of the Kohn
conductivity in which the change of the
Hamiltonian upon the introduction of the complex
flux is measured. This code is intended for
usage both in the interacting and the
noninteracting cases. In the current version,
we only allow calculations with models that
contain the suffix '_complex' in their
module name.

"""
import sys
import numpy as np

from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

from _kohn_diag import kohn_diag

save_metadata = True
allowed_modules = ['anderson', 'anderson_complex', 'heisenberg_complex']

if __name__ == '__main__':

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    if model_name not in allowed_modules:

        print((f'Calculation for module {model_name} not supported! '
               f'Allowed modules are {allowed_modules}. Exiting!'))
        sys.exit()

    # if we are not dealing with an anderson-type model
    # then this is an interacting case in 1D!
    complex_ = False
    interacting = False
    if 'anderson' not in model_name:
        interacting = True

    if 'complex' in model_name:

        complex_ = True

    if interacting:
        # make sure not both
        # complex factors are nonzero
        # we allow for two equivalent formulations
        # of the problem that only differ by a
        # gauge transformation.
        print('Kohn calculation, interacting case message.')
        if ((argsDict['J_phase'] != 0.) and (argsDict['phase_bc'] != 0.0)):
            print(('Please make sure not both phase factors are nonzero. '
                   'Exiting.'))
            sys.exit()
        if ((argsDict['J_phase'] == 0.) and (argsDict['phase_bc'] == 0.0)):
            print(('WARNING! Both complex phases are set to zero! '
                   'Thouless conductivity calculation will be performed!'))
            complex_ = False
    else:
        phase_keys = [key for key in argsDict.keys() if 'phase' in key]

        phase_factor_ = {key: argsDict[key]
                         for key in phase_keys if argsDict[key] != 0.}
        if not phase_factor_:
            print('Kohn calculation, noninteracting case message')
            print(('WARNING: all complex phases are set to zero! '
                   'Thouless conductivity calculation will be performed!'))
            complex_ = False

    if argsDict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsDict["pbc"]} which is not '
               'supported. Exiting.'))
        sys.exit()

    for seed in range(min_seed, max_seed + 1):
        # perform the first part of the calculation
        # as is
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed

        eigvals_dict, argsDict = kohn_diag(
            mod, argsDict, interacting, complex_,)

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
