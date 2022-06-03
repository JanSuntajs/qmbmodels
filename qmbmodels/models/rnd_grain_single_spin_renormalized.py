"""
This module contains tools for construction
of the model simulating an ergodic grain coupled
to a single localized LIOM (local integral of motion),
or, in plain terms, a single spin. In this implementation,
however, we make sure the Hilbert-Schmidt norms of the operators
are properly normalized and thus do not scale with the
system size. That makes the comparison of different
Hamiltonian terms more tractable.


The model is inspired by the one used in the
following paper:
https://arxiv.org/abs/1705.10807
and is also a variation of the one implemented
in the rnd_grain module. Technically, one could
achieve the same effect also using the rnd_grain.py
module, but I've decided to provide a separate module
to avoid confusion regarding the parameter specification.
Also, this model does not store the grain random numbers
to save space. For large grains, the number of random
values to be stored can be significant.

The model is as follows:

H = R + g0*alpha^{x_L} * S^x_{j_i} * S^x_L + 
    h_L * S^z_L


Where L_b is the length of the chain modeled by an ergodic grain
and L_loc is the length of the localized part, the full length L
thus being equal to L = L_b + L_loc. In python's notation, we place
the grain between the zeroth and (L_b - 1)-th site, including both
ends. We do not allow for periodic boundary conditions in this model.

Where R is a symmetric random matrix defined as

    R = (1/2) * (A + A.T) * (1 / 2**L_b)

With matrix elements A_{ij} being distributed normally with a zero
mean and a standard deviation of \sqrt{2}. The last multiplicative
factor is to ensure the Hilbert-Schmidt norm is equal to unity.

The second term models the coupling of the grain to the LIOMs, where
    
    alpha = exp(-1./xi),

with xi being the LIOM's localization length and g0 controls the interaction
strength. Note the parameter x_i in the exponent of alpha, which denotes
the distance of the i-th LIOM from the bath. We have devised this model
essentialy as a 0-d model, since the LIOMs do not interact with each other.
To achieve the most generic behavior for such a toy model of the MBL transition
in 0-d, we sample the lengths x_i according to some probability distribution,
in this case, the uniform one on the interval between 1 (we fix the minimum distance
to the grain) and L_loc.

The operator S^x_{j_i} is a bath operator, where the subscript
i_j denotes a randomly chosen site within the bath to which the i-th LIOM
couples.

The last term models the random fields acting on the LIOMs in the localized part.

Further implementations of the model might include setups with the grain being
immersed into surounding localized regions to see whether any additional scaling
factors emerge in the finite size scaling analysis as a consequence of two borders. 
"""

import numpy as np

from ham1d.models.spin1d import hamiltonian as sphm

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'L_b'] + comm_syspar_keys
modpar_keys = ['g0', 'alpha', 'W', 'dW'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def _create_grain(size, seed):
    """
    Create the random grain matrix in a reproducible
    way.

    Parameters:

    beta: the grain strength parameter

    size: grain size

    seed: random seed

    returns:

    mat: 2**size x 2**size random matrix
    """
    rng = np.random.default_rng(seed)

    mat = rng.normal(0, np.sqrt(2), (2**size, 2**size))

    return (0.5) * (mat + mat.T)




def _disorder_dist(size, seed, W=0.5):
    """
    A routine for creating the disorder
    distribution used in the paper.

    """

    rng = np.random.default_rng(seed)

    return rng.uniform(1 - W, 1 + W, size)


def _coupling_dists(size_bath, size_loc, seed):
    """
    A routine for creating a list of indices
    within the bath to which the LIOMs couple.


    """
    rng = np.random.default_rng(seed)
    bath_indices = np.arange(size_bath)

    return rng.choice(bath_indices, size_loc)


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0, dtype=np.float64):
    """


    """

    # length of the bath and
    # the localized part
    L = argsdict['L']
    L_b = argsdict['L_b']

    L_loc = int(L - L_b)


    disorder = argsdict['disorder']
    beta = argsdict['beta']

    alpha = argsdict['alpha']
    g0 = argsdict['g0']

    if L_b < 1:
        raise ValueError(('Bath length L_b should '
                          'be greater than 0!'))

    # no pbc by design hereo
    if argsdict['pbc']:

        raise ValueError('No pbc allowed in this model!')

    ham_type = argsdict['ham_type']

    if ham_type != 'spin1d':

        raise ValueError('Only spin1d ham_type allowed here!')

    ham = sphm
    seed = argsdict['seed']

    # grain is of size L - 1 here!
    rnd_grain = _create_grain(L - 1, seed)

    coupling_indices = _coupling_dists(L - 1, 1, seed)
    fields = get_disorder_dist(1, disorder, argsdict['W'],
                               argsdict['dW'], seed)

    grain_term = ['RR', [[1./np.sqrt(2**L_b), 0, L - 2]]]

    static_list = [grain_term]
    grain_list = [rnd_grain]


    bath_couplings = [[g0 * alpha**L_loc, coupling_indices[0], L - 1, ]]

    bath_loc_term = ['xx', bath_couplings]

    rnd_term = ['z', [[fields[0], L - 1]]]

    static_list += [bath_loc_term, rnd_term]

    hamiltonian = ham(L, static_list, [], Nu=None,
                      grain_list=grain_list, parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize,
                      dtype=dtype)

    return hamiltonian, {
                         'Hamiltonian_random_disorder': fields,
                         'Hamiltonian_bath_coupling_indices_disorder':
                         coupling_indices}
