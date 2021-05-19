"""
This module contains the
tools for construction of
the Anderson nonintearacting
model in system of arbitrary
(integer) dimensionality in
the complex arithmetic which allows
for arbitrary phase factors to
be obtained when traversing the
system's boundary. In contrast
to the anderson_complex module,
this module is intended for the
analysis of the effects of flux on
the Hamiltonian's eigenlevels at
a fixed realization of disorder.
Instead of changing different disorder
realizations, different seeds thus
increase the flux values. Also note that
the model is expected to have periodic
boundary conditions across dim-1 axes
and complex bc along the remaining axis.
The module is intended for changing the
phase between 0 and pi.

Attributes:
-----------

syspar_keys: list

modpar_keys: list

"""


import numpy as np
import sys

from anderson.model import hamiltonian as ham

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'dim', 'disorder_switch', 'phase_nvals', 'logspace'] + comm_syspar_keys
# disorder switch can be used to control different disorder
# realizations
# phase_n is used to control the number of phases to be analysed on
# the interval between 0 and pi
modpar_keys = ['t', 'W', 'dW'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict_, parallel=False, mpirank=0, mpisize=0):
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
    argsdict = argsdict_.copy()
    L = argsdict['L']
    dim = argsdict['dim']

    # phase values for which to perform the sweep
    n_steps = argsdict['phase_nvals']
    if argsdict['logspace']:
        print('logspace True')
        phase_vals = np.logspace(-4, np.log10(np.pi), n_steps)
    else:
        print('logspace_false')
        phase_vals = np.linspace(0., np.pi, n_steps)

    print(f'phase_vals: {phase_vals}')

    seed = argsdict['seed']
    # make sure seed is not greater than the number of
    # steps
    if ((seed >= n_steps) or (seed < 0)):
        print(('Seed is greater than the number of steps '
               'or smaller than zero! Exiting.'))
        sys.exit()

    # cast the pbc in the appropriate form for the task
    if argsdict['pbc'] is not True:
        print(('Please set the pbc parameter '
               'equal to True. Now you have '
               f'{argsdict["pbc"]} which is not '
               'supported. Exiting.'))
        sys.exit()
    #pbc = np.copy(argsdict['pbc'])
    # set up equal to one, then change them along one axis
    pbc_ = np.array(
        [1. + 0j for i in range(argsdict['dim'])], dtype=np.complex128)
    pbc_[-1] = np.exp(1j * phase_vals[seed])

    hopping = argsdict['t']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if ham_type != 'anderson':

        err_message = ('ham_type {} not allowed for the'
                       'weak links model! Only anderson '
                       'case is valid!').format(ham_type)
        raise ValueError(err_message)

    fields = get_disorder_dist(L, disorder, argsdict['W'],
                               0.5 * argsdict['dW'],
                               int(argsdict['disorder_switch']),
                               dim=dim)

    hamiltonian = ham(L, dim, hopping, fields, pbc_, parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize, dtype=np.complex128)

    return hamiltonian, {'Hamiltonian_random_disorder': fields.flatten(),
                         'Hamiltonian_random_phase': pbc_}
