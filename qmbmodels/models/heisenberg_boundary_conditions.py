"""
This module contains the tools for construction
of the Heisenberg XXZ model with nearest and
neighbour terms and the possibility
to also include the anti-periodic boundary
conditions. For now, this model repeats code
of the Heisenberg model and was created to avoid
pollution of the other currently existing modules
which were not intended for calculations
in the anti-periodic case. This shall hopefully
change in the future. 

The model in the hard-core boson (spin) case:

H = 0.5 * J1 * \sum_i (s_i^+ s_{i+1}^- + s_i^- s_{i+1}^-) +
    0.5 * J2 * \sum_i (s_i^+ s_{i+2}^- + s_i^- s_{i+2}^-) +
    delta1 * J1 * \sum_i s_i^z s_{i+1}^z +
    delta2 * J2 * \sum_i s_i^z s_{i+2}^z +
    \sum_i w_i s_i^z

The model in the fermionic case reads:

H = 0.5 * J1 * \sum_i (c_i^+ c_{i+1}^- + c_{i+1}^+ c_i^-) +
    0.5 * J2 * \sum_i (c_i^+ c_{i+2}^- + c_{i+2}^+ c_i^-) +
    delta1 * J1 * \sum_i n_i^z n_{i+1}^z +
    delta2 * J2 * \sum_i n_i^z n_{i+2}^z +
    \sum_i w_i n_i

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
    'J', 'delta', 'W', 'dW': float
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
modpar_keys = ['J', 'delta',
               'W', 'dW'] + comm_modpar_keys

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

    # create an array of pbc factors -> ones for
    # pbc or obc (lengths differ in the two cases)
    # or with -1 entries on appropriate sites
    pbc_1 = np.ones_like(range(L))
    pbc_2 = np.ones_like(range(L))
    pbc_3 = np.ones_like(range(L))
    # antiperiodic bc -> first an last
    # coupling get a -1 -> a phase factor of pi
    if pbc == -1.:
        print('antiperiodic bc')
        pbc_1[-1] = -1.
        pbc_2[0] = -1.

    couplings_1 = 0.5 * argsdict['J'] * pbc_1
    couplings_2 = 0.5 * argsdict['J'] * pbc_2

    if ham_type == 'ferm1d':

        ham = fehm

        hops = [
            ['+-', [[couplings_1[i], *coup] for i, coup
                    in enumerate(coup1p)]],
            ['+-', [[couplings_2[i], *coup] for i, coup
                    in enumerate(coup1m)]],
        ]
        num_op = 'n'

    elif ham_type == 'spin1d':

        ham = sphm
        hops = [['+-', [[couplings_1[i], *coup] for i, coup
                        in enumerate(coup1p)]],
                ['-+', [[couplings_2[i], *coup] for i, coup
                        in enumerate(coup1p)]], ]

        num_op = 'z'

    inter = [[num_op + num_op, [[argsdict['J'] * argsdict['delta'] * pbc_1[i],
                                 *coup]
                                for i, coup in enumerate(coup1p)]]]

    fields = get_disorder_dist(L, disorder, argsdict['W'],
                               argsdict['dW'], argsdict['seed'])

    print('hops')
    print(hops)
    rnd = [num_op, [[field, i] for i, field in enumerate(fields)]]

    static_list = [*hops, *inter, rnd]

    hamiltonian = ham(L, static_list, [], Nu=int(nu), parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    return hamiltonian, {'Hamiltonian_random_disorder': fields}
