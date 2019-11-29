"""
This module contains a routine
that selects a proper hamiltonian
module to import in the calling
main script.

"""
import sys
from utils.cmd_parser_tools import arg_parser_general


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
