"""
This code is intended for simulating the
"weak-link Heisenberg model" where the
exchange couplings in the model are
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

System parameters:

L: int
   system size
nu: int
    number of up spins
J: float
   J_max, the upper boundary for the J values.
   Set to 1. in the above paper.
lambda_: float
   Parameter controlling the supposed
   diffusive or subdiffusive behaviour of the
   system.
W: float
   Potential disorder
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

    L = argsdict['L']
    nu = argsdict['nu']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if ham_type == 'spin1d':

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

    else:

        print(('Module {} not allowed! Only spin1d works for the '
               'heisenberg_weak_links '
               'model!')
              .format(ham_type))

        return None
