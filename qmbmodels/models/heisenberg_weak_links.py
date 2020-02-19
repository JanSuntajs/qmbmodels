"""
This module contains the tools for construction
of the "weak-link Heisenberg model".

The exchange couplings in the model are
distributed randomly according to some
distribution. In the implementation we
consider the following PRL paper
by M. Kozarzewski, P. Prelovsek and M.
Mierzejewski:

Spin subdiffusion in the Disordered Hubbard
chain:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.246602

Received 26 March 2018
DOI:https://doi.org/10.1103/PhysRevLett.120.246602

The model is defined as follows:
H = \sum_i J_i S_i * S_{i+1}
where S_i are the spin operators and J_i
are randomly distributed disorder values where the
disorder is distributed as:

f(J) = lambda_ * J ** (lambda_ - 1)
J in [0, J_max]; note: in the above paper, J_max = 1

In order to simulate this distribution using
the random uniform distribution on an interval
[0, 1], we use the following relation:

J = x ** (1 / lambda_); x in [0, 1]

Note: this model is only implemented for the spin1d
case.

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
    weak-link model:
    J, lambda, W
    Width of the J-disorder distribution,
    power-law distribution exponent,
    width of the random potential noise
    distribution, respectively.
    The rest: see the docstring for the _common_keys
    module.
"""


import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm


from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'nu'] + comm_syspar_keys
modpar_keys = ['J', 'lambda', 'W'] + comm_modpar_keys

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
        'nu': int, number of up spins
        'pbc': bool, True for periodic boundary conditions, False
               for open boundary conditions
        'disorder': str, specifies the disordered potential used.
        'ham_type': str, which Hamiltonian type is used ('spin1d'
                    for spin Hamiltonians, 'ferm1d' for fermionic
                    ones and 'free1d' for free models).
        'J', 'lambda', 'W': float
                    Width of the J-disorder distribution,
                    power-law distribution exponent,
                    width of the random potential noise
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
    nu = argsdict['nu']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    # if ham_type == 'spin1d':
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

    # prepare the random field distributions -> only the powerlaw
    # dist is allowed for this particular model
    J_fields = get_disorder_dist(len(coup), 'powerlaw', argsdict['lambda'],
                                 argsdict['J'], argsdict['seed'])

    pm = [['xx', [[J_fields[i], *inter]
                  for i, inter in enumerate(coup)]]]

    mp = [['yy', [[J_fields[i], *inter]
                  for i, inter in enumerate(coup)]]]

    zz = [['zz', [[J_fields[i], *inter]
                  for i, inter in enumerate(coup)]]]

    rnd_noise = np.random.uniform(-argsdict['W'],
                                  argsdict['W'], size=L)
    z_loc = ['z', [[rnd_noise[i], i] for i in range(L)]]
    static_list = [*pm, *mp, *zz, z_loc]

    hamiltonian = ham(L, static_list, [], Nu=int(nu), parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    fields = {'Hamiltonian_J_random_disorder': J_fields,
              'Hamiltonian_W_random_disorder': rnd_noise}

    return hamiltonian, fields
