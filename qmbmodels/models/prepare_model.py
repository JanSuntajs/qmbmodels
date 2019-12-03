"""
This module contains a routine
that selects a proper hamiltonian
module to import in the calling
main script.

"""
import sys
from utils.cmd_parser_tools import arg_parser_general
from utils.cmd_parser_tools import arg_parser
from models._common_keys import minmax_seed_default


def select_model():
    """
    This function is called in the main running scripts
    to determine which hamiltonian should be used
    base on the provided command-line arguments.

    returns
    -------

    mod: python module with a Hamiltonian model definition

    model['model']: the name of the module with the definition
    """

    model, tmp = arg_parser_general({'model': [str, '']})

    if model['model'] == 'heisenberg':
        from models import heisenberg as mod
    elif model['model'] == 'imbrie':
        from models import imbrie as mod
    else:
        print(('model {} not '
               'recognised! Exiting!').format(model['model']))
        sys.exit(0)
    print('Using model {}'.format(model['model']))

    return mod, model['model']


def import_model(model):
    """
    This function is mainly intended for
    usage in the preparation scripts, such
    as the runner.py - like scripts, where
    it imports an appropriate hamiltonian
    model so that the appropriate modpar
    and syspar keys can be used.

    """

    if model == 'heisenberg':
        from models import heisenberg as mod
    elif model == 'imbrie':
        from models import imbrie as mod
    else:
        print(('model {} not '
               'recognised! Exiting!').format(model))
        sys.exit(0)

    return mod


def get_module_info():
    """
    Obtain the system and model parameter
    keys relevant for the model used.

    Parameters:
    -----------
    None

    Returns:
    --------

    mod: python module containing the defs.
         of the used hamiltonian (for instance,
         imbrie.py or heisenberg.py)
    model_name: str
         the name corresponding to the used
         python module defining the model
         hamiltonian.
    argsDict: dict
         Dictionary containing formatted pairs
         of parameter keys and their corresponding
         values which are parsed from the command-line
         arguments.
    seedDict: dict
         Dictionary containing the keys corresponding
         to the range of random seed values we should
         be iterating over when performing different
         disorder realizations.
         The seedDict should have the following keys:
         'min_seed', indicating the starting seed
         number and 'num_seed', indicating the number
         of seed values we should be using. The maximum
         seed number is determined as:
         seedDict['min_seed'] + seedDict['num_seed']
    syspar_keys: list
         A list of strings indicating which of the keys
         in the argsDict correspond to the system parameters.
    modpar_keys: list
         A list of strings indicating which of the keys in the
         argsDict correspond to the model parameters.
    savepath: string
         Path to the root of the storage forlder where one
         wants to store the results.
    syspar: string
         A formatted string containing information about all
         the relevant system parameters.
    modpar: string
         A formatted string containing information about all
         the relevant model parameters.
    minseed, maxseed: int
         Integers indicating the minimum and maximum seed
         numbers. See above for their definitions.
    """

    mod, model_name = select_model()
    syspar_keys = mod.syspar_keys
    modpar_keys = mod._modpar_keys

    argsDict, extra = arg_parser(syspar_keys, modpar_keys)
    syspar_keys.append('model')
    argsDict['model'] = model_name

    seedDict, extra = arg_parser_general(minmax_seed_default)

    minseed = seedDict['min_seed']
    maxseed = seedDict['max_seed']
    stepseed = seedDict['step_seed']

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    return (mod, model_name, argsDict, seedDict,
            syspar_keys, modpar_keys, savepath,
            syspar, modpar, minseed, maxseed)
