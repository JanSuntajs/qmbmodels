"""
This module contains the
tools for construction of
the Anderson nonintearacting
model in system of arbitrary
(integer) dimensionality in
the complex arithmetic which allows
for arbitrary phase factors to
be obtained when traversing the
system's boundary.

Attributes:
-----------

syspar_keys: list

modpar_keys: list

"""


import numpy as np

from anderson.model import hamiltonian as ham

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'dim'] + comm_syspar_keys
modpar_keys = ['t', 'W', 'dW', 'boundary_phase'] + comm_modpar_keys

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
        'boundary_phase': complex
            0 or a complex number with a unit modulus specifying
            the phase obtained when traversing the system boundary.

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
    dim = argsdict['dim']

    pbc = argsdict['pbc']

    hopping = argsdict['t']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if ham_type != 'anderson':

        err_message = ('ham_type {} not allowed for the'
                       'weak links model! Only anderson '
                       'case is valid!').format(ham_type)
        raise ValueError(err_message)

    fields = get_disorder_dist(L, disorder, argsdict['W'],
                               0.5 * argsdict['dW'], argsdict['seed'], dim=dim)

    hamiltonian = ham(L, dim, hopping, fields, pbc, parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize, dtype=np.complex128)

    return hamiltonian, {'Hamiltonian_random_disorder': fields.flatten()}
