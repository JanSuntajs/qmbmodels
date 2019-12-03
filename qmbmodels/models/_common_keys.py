"""
This module defines
some syspar and modpar
keys that are used commonly
in all models (imbrie, heisenberg
etc.)

"""

comm_syspar_keys = ['pbc', 'disorder', 'ham_type']

comm_modpar_keys = ['min_seed', 'num_seed']

minmax_seed_default = {'min_seed': [int, 0],
                       'num_seed': [int, 0]}
