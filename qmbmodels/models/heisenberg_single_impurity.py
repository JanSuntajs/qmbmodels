"""
This module contains the tools for construction
of the Heisenberg XXZ model with nearest neighbours
and a single impurity at the middle of the chain. In
order to break the reflection symmetry, some small
noise (small constant potential) is added to the
first site in the chain.

Attributes:
-----------

syspar_keys: list
    System parameter descriptors.
    L: int, system size
    nu: int, number of up spins
    The rest: see the docstring for
    the _common_keys module.

modpar_keys: list
    Module parameters for the
    Heisenberg model:
    'J1', 'J2', 'delta1', 'delta2', 'W', 'dW': float
    nearest and next nearest echange couplings, nearest
    and next-nearest anisotropy parameters, center of
    the disorder distribution and width of the disorder
    distribution, respectively.
    The rest: see the docstring for the _common_keys
    module.
"""


import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm
from ham1d.models.ferm1d import hamiltonian as fehm

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'nu'] + comm_syspar_keys
modpar_keys = ['J1', 'delta1',
               'W', 'dW', 'noise'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0):
    """
    Constructs the Heisenberg's XXZ model Hamiltonian with
    nearest and next-nearest interaction and exchange terms.

    Parameters:
    -----------

    argsdict: dict
        A dictionary containing the information needed to construct
        the Hamiltonian. Should have the following keys:
        'L': int, system size
        'nu': int, number of up spins
        'pbc': bool, True for periodic boundary conditions, False
               for open boundary conditions
        'disorder': str, specifies the disordered potential used.
        'ham_type': str, which Hamiltonian type is used ('spin1d'
                    for spin Hamiltonians, 'ferm1d' for fermionic
                    ones and 'free1d' for free models).
        'J1', 'J2', 'delta1', 'delta2', 'W', 'dW': float
            Model parameters for the Heisenberg model: nearest
            and next nearest echange couplings, nearest and
            next-nearest anisotropy parameters, center of
            the disorder distribution and width of the disorder
            distribution, respectively.

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
        ham1d package

    dict:
        A dictionary containing the potential disorder
        under the key 'Hamiltonian_random_disorder'.

    """

    L = argsdict['L']
    nu = argsdict['nu']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    if disorder != 'single':

        raise ValueError(('This model is only designed to work with '
                          'the single impurity model!'))

    ham_type = argsdict['ham_type']

    if pbc:
        range_1 = range_2 = range(L)
        coup1p = [[i, (i + 1) % L] for i in range_1]
        coup1m = [[i, (i - 1) % L] for i in range_1]

    else:
        range_1 = range(L - 1)
        range_2 = range(1, L)

        coup1p = [[i, (i + 1)] for i in range_1]
        coup1m = [[i, (i - 1)] for i in range_2]

    if ham_type == 'ferm1d':

        ham = fehm

        hops = [['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]],
                ['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1m]],
                ]
        num_op = 'n'

    elif ham_type == 'spin1d':

        ham = sphm
        hops = [['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]],
                ['-+', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]]]

        num_op = 'z'

    inter = [[num_op + num_op, [[argsdict['J1'] * argsdict['delta1'],
                                 *coup]
                                for coup in coup1p]], ]
    # put the impurity at the center of the chain
    fields = get_disorder_dist(L, disorder, argsdict['W'],
                               argsdict['dW'], argsdict['seed'], loc=int(L * 0.5))

    # put random noise on the first spot
    fields[0] = argsdict['noise']

    rnd = [num_op, [[field, i] for i, field in enumerate(fields)]]

    static_list = [*hops, *inter, rnd]

    hamiltonian = ham(L, static_list, [], Nu=int(nu), parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    return hamiltonian, {'Hamiltonian_random_disorder': fields}
