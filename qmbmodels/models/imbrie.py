"""
This module contains the tools for construction
of the Imbrie model Hamiltonian.

For more details, see the following paper by
John Z. Imbrie (arXiv: 1403.7837v4 [math-ph]
30 Mar 2016):
https://arxiv.org/pdf/1403.7837.pdf

The model:
H = \sum_i h_i S_i^z + \sum_i \gamma_i S_i^x + \
    \sum_i J_i S_i^z S_{i+1}^z

Attributes:
-----------

syspar_keys: list
    System parameter descriptors.
    L: int, system size

modpar_keys: list
    Module parameters for the
    Heisenberg model:
    'J', 'dJ', 'H', 'dH', 'Gamma', 'dGamma': float
    Center and width of the distribution for the
    random echange couplings (the Ising
    exchange term), center and width of the
    distribution for the random potentials in the
    z-direction (sigma z term)
    and center and width of the distribution
    for the potentials in the x-direction (sigma x term).
    The rest: see the docstring for the _common_keys
    module.
"""

import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L'] + comm_syspar_keys
modpar_keys = ['J', 'dJ', 'H', 'dH',
               'Gamma', 'dGamma'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0):
    """
    Constructs the "Heisenberg weak link" model Hamiltonian with
    random J-exchange and random potential noise.

    Parameters:
    -----------

    argsdict: dict
        A dictionary containing the information needed to construct
        the Hamiltonian. Should have the following keys:
        'L': int, system size
        'pbc': bool, True for periodic boundary conditions, False
               for open boundary conditions
        'disorder': str, specifies the disordered potential used.
        'ham_type': str, which Hamiltonian type is used ('spin1d'
                    for spin Hamiltonians, 'ferm1d' for fermionic
                    ones and 'free1d' for free models).
        'J', 'dJ', 'H', 'dH', 'Gamma', 'dGamma': float
                    Center and width of the distribution for the
                    random echange couplings (the Ising
                    exchange term), center and width of the
                    distribution for the random potentials in the
                    z-direction (sigma z term)
                    and center and width of the distribution
                    for the potentials in the x-direction
                    (sigma x term).

    parallel: boolean, optional
        Whether the Hamiltonian is to be constructed in parallel
        or not. Defaults to False.

    mpirank: int, optional
        In case when parallel==True (e.g. when mpi
        parallel construction is used), specifies the mpi process
        used to construct the parallel block of the Hamiltonian.
        Defaults to 0.

    mpisize: int, optional
        In case mpi parallelism is used, specifies the size of
        the mpi pool. Defaults to 0 which corresponds to the
        sequential case.

    Returns:
    --------

    hamiltonian:
        An instance of the hamiltonian class from the
        ham1d package. This routine only works for the
        spin1d case.

    dict:
        A dictionary containing the potential disorder
        under the key 'Hamiltonian_random_disorder'.

    Raises:
    -------

    ValueError
        If ham_type is not spin1d.

    """

    L = argsdict['L']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if ham_type != 'spin1d':

        err_message = ('ham_type {} not allowed for the'
                       'weak links model! Only spin1d '
                       'case is valid!').format(ham_type)
        raise ValueError(err_message)

    if pbc:

        coup = [[i, (i + 1) % L] for i in range(L)]

    else:

        coup = [[i, (i + 1)] for i in range(L - 1)]

    ham = sphm

    # prepare the random field distributions
    J_fields = get_disorder_dist(len(coup), disorder, argsdict['J'],
                                 argsdict['dJ'], argsdict['seed'])
    H_fields = get_disorder_dist(L, disorder, argsdict['H'],
                                 argsdict['dH'], argsdict['seed'] * 2)
    Gamma_fields = get_disorder_dist(L, disorder, argsdict['Gamma'],
                                     argsdict['dGamma'],
                                     argsdict['seed'] * 3)

    ising = [['zz', [[J_fields[i], *inter]
                     for i, inter in enumerate(coup)]]]
    # ising = [['zz', [[J_fields[i], *inter] for i, inter in
    #                  enumerate(coup)]]

    rnd_z = ['z', [[field, i] for i, field in enumerate(H_fields)]]
    rnd_x = ['x', [[field, i] for i, field in enumerate(Gamma_fields)]]

    static_list = [*ising, rnd_x, rnd_z]

    hamiltonian = ham(L, static_list, [], Nu=None, parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    fields = {'Hamiltonian_J_random_disorder': J_fields,
              'Hamiltonian_H_random_disorder': H_fields,
              'Hamiltonian_Gamma_random_disorder': Gamma_fields}
    return hamiltonian, fields

