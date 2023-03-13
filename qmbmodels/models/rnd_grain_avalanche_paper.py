"""
This module reimplements the quantum sun model found
in the rnd_grain.py module, such that it is compatible
with the definitions we used in our follow-up paper
on the quantum avalanches. 

The original model was inspired by the one that can
be found in the link below:

https://arxiv.org/abs/1705.10807

The model is as follows:

H = H_dot + \\sum_{i=L_b + 1}^{L_b + L_loc} g0*alpha^{x_i} * S^x_{j_i} * S^x_i + 
    \\sum_{i=L_b + 1}^{L_b + L_loc} h_i * S^z_i


Where L_b is the length of the chain modeled by an ergodic grain
and L_loc is the length of the localized part, the full length L
thus being equal to L = L_b + L_loc. In python's notation, we place
the grain between the zeroth and (L_b - 1)-th site, including both
ends. We do not allow for periodic boundary conditions in this model.

Above, H_dot is defined as follows:

    H_dot = \gamma * R / \frac{2^N + 1}

Here, R describes the properties of a random dot, where R is drawn
from a Gaussian orthogonal ensemble (GOE), 
R = \frac{1}{\sqrt{2}}(A + A^T); the matrix elements A_{ij} are
drawn from a normal distribution with zero mean and unit variance.
This is the main source of difference with the previous implementation
of the model. Note that we could have simply solved this by properly
rescaling the parameter beta in the original formulation of the model,
however, we chose to define a new model to avoid confusion during the
later stages of our project.

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

syspar_keys = ['L', 'L_b', 'dist_type'] + comm_syspar_keys
modpar_keys = ['gamma', 'g0', 'alpha', 'W', 'dW', 'epslen'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def _create_grain(size, seed):
    """
    Create the random grain matrix in a reproducible
    way. Here, we define the R matrix as:

        R = (\frac{1.}{np.sqrt{2.}}) (A + A^T)

    Parameters:

    beta: the grain strength parameter

    size: grain size

    seed: random seed

    returns:

    mat: 2**size x 2**size random matrix
    """
    rng = np.random.default_rng(seed)

    mat = rng.normal(0, 1, (2**size, 2**size))

    return (1./np.sqrt(2.)) * (mat + mat.T)


def _length_dist(size, seed, dist_type=0, *args, **kwargs):
    """
    A routine for creating distributions
    having the same mean as the discrete
    distribution used in Luitz's paper:
    (1/L)*sum_{i=1}^L i = (L + 1)/2

    """
    rng = np.random.default_rng(seed)
    if dist_type == 0:
        return np.sort(rng.uniform(1, size, size))
    else:

        if dist_type == 1:

            sites = np.arange(1, size + 1, dtype=np.float64)
            sites += rng.uniform(-kwargs['eps'], kwargs['eps'], size)
        if dist_type == 2:

            sites = np.arange(0, size, dtype=np.float64)

            sites[1:] += rng.uniform(-kwargs['eps'], kwargs['eps'], size - 1)

        return sites


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


def construct_hamiltonian(argsdict, parallel=False, mpirank=0,
                          mpisize=0, dtype=np.float64):
    """
    Docstring here

    """

    # length of the bath and
    # the localized part
    L = argsdict['L']
    L_b = argsdict['L_b']

    L_loc = int(L - L_b)

    len_dist_type = argsdict['dist_type']

    disorder = argsdict['disorder']
    gamma = argsdict['gamma']

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

    rnd_grain = _create_grain(L_b, seed)
    lengths = _length_dist(L_loc, seed, len_dist_type, eps=argsdict['epslen'])
    coupling_indices = _coupling_dists(L_b, L_loc, seed)
    fields = get_disorder_dist(L_loc, disorder, argsdict['W'],
                               argsdict['dW'], seed)

    grain_term = ['RR', [[gamma / np.sqrt(2. ** L_b + 1), 0, L_b - 1]]]

    static_list = [grain_term]
    grain_list = [rnd_grain]

    if L_loc > 0:

        bath_couplings = [[g0 * alpha**length, coupling_indices[i], L_b + i, ]
                          for i, length in enumerate(lengths)]
        bath_loc_term = ['xx', bath_couplings]

        rnd_term = ['z', [[fields[j], i]
                          for j, i in enumerate(range(L_b, L))]]

        static_list += [bath_loc_term, rnd_term]

    hamiltonian = ham(L, static_list, [], Nu=None,
                      grain_list=grain_list, parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize,
                      dtype=dtype)

    return hamiltonian, {'Hamiltonian_grain_matrix_disorder':
                         rnd_grain.flatten(),
                         'Hamiltonian_random_disorder': fields,
                         'Hamiltonian_lengths_disorder': lengths,
                         'Hamiltonian_bath_coupling_indices_disorder':
                         coupling_indices}
