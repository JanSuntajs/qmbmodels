import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm
from ham1d.models.ferm1d import hamiltonian as fehm

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'nu'] + comm_syspar_keys
modpar_keys = ['J1', 'J2', 'delta1', 'delta2',
               'W', 'dW'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0):

    L = argsdict['L']
    nu = argsdict['nu']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if pbc:
        range_1 = range_2 = range_3 = range_4 = range(L)
        coup1p = [[i, (i + 1) % L] for i in range_1]
        coup1m = [[i, (i - 1) % L] for i in range_1]
        coup2p = [[i, (i + 2) % L] for i in range_1]
        coup2m = [[i, (i - 2) % L] for i in range_1]

    else:
        range_1 = range(L - 1)
        range_2 = range(L - 2)
        range_3 = range(1, L)
        range_4 = range(2, L)

        coup1p = [[i, (i + 1)] for i in range_1]
        coup1m = [[i, (i - 1)] for i in range_3]
        coup2p = [[i, (i + 2)] for i in range_2]
        coup2m = [[i, (i - 2)] for i in range_4]

    if ham_type == 'ferm1d':

        ham = fehm

        hops = [['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]],
                ['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1m]],
                ['+-', [[0.5 * argsdict['J2'], *coup] for coup in coup2p]],
                ['+-', [[0.5 * argsdict['J2'], *coup] for coup in coup2m]],
                ]
        num_op = 'n'

    elif ham_type == 'spin1d':

        ham = sphm
        hops = [['+-', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]],
                ['-+', [[0.5 * argsdict['J1'], *coup] for coup in coup1p]],
                ['+-', [[0.5 * argsdict['J2'], *coup] for coup in coup2p]],
                ['-+', [[0.5 * argsdict['J2'], *coup] for coup in coup2p]]]

        num_op = 'z'

    inter = [[num_op + num_op, [[argsdict['J1'] * argsdict['delta1'],
                                 *coup]
                                for coup in coup1p]],
             [num_op + num_op, [[argsdict['J2'] * argsdict['delta2'],
                                 *coup]
                                for coup in coup2p]]]

    fields = get_disorder_dist(L, disorder, argsdict['W'],
                               argsdict['dW'], argsdict['seed'])

    rnd = [num_op, [[field, i] for i, field in enumerate(fields)]]

    static_list = [*hops, *inter, rnd]

    hamiltonian = ham(L, static_list, [], Nu=int(nu), parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    return hamiltonian, {'Hamiltonian_random_disorder': fields}
