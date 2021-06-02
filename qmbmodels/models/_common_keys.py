"""
Specifies which system and module
parameter keys are used in all the
physical models.

Attributes:
-----------

comm_syspar_keys: list
    Which system parameter descriptor keys are common
    across all the models used.

    All the models make use of the information
    about the validity of the periodic boundary
    conditions ('pbc'), type of the disorder ('disorder')
    and type of the hamiltonian used ('ham_type', can
    be 'spin1d', 'ferm1d', 'free1d').

comm_modpar_keys: list
    Which module parameter keys are common across all
    the models used.

    The contents of this list refer to the seeds used
    in the generation of random potentials used in the
    Hamiltonians.

minmax_seed_default: dict
    Provides default values for the values
    corresponding to the comm_modpar_keys.

"""

comm_syspar_keys = ['pbc', 'disorder', 'ham_type', 'save_space']

comm_modpar_keys = ['min_seed', 'max_seed', 'step_seed']

minmax_seed_default = {'min_seed': [int, 0],
                       'max_seed': [int, 0],
                       'step_seed': [int, 0]}
