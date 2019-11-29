import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm

from .disorder import get_disorder_dist

syspar_keys = ['L', 'pbc', 'disorder', 'ham_type']
modpar_keys = ['J', 'dJ', 'H', 'dH',
               'Gamma', 'dGamma', 'min_seed', 'max_seed']

_modpar_keys = [key for key in modpar_keys if 'seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0):

    L = argsdict['L']

    pbc = argsdict['pbc']
    disorder = argsdict['disorder']
    ham_type = argsdict['ham_type']

    if ham_type == 'spin1d':

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

    else:

        print('Module {} not allowed! Only spin1d works for the Imbrie model!'
              .format(ham_type))

        return None
