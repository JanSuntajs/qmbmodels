"""

ALTERNATIVE DEFINITION, TRYING TO GET 
A SPEEDUP
To avoid code repetition, we implement
the routines needed in the calculation
of the many-body entanglement entropy
for the (any-dimensional) Anderson model
in a separate module displayed here.


"""


import numpy as np
import numba as nb
from numba.types import bool_
from numpy.random import default_rng
from scipy.special import comb
from scipy.linalg import eigh
from itertools import permutations

from anderson.model import hamiltonian as ham
from anderson.operators import get_idx, get_coordinates


# a note about some naming conventions:
#
# partition_fraction: how big the subpartition
# is w.r.t. the whole system
#
# particle_filling: probability of finding a single-body
# state occupied or empty
#


# a helper routine for an easier calculation
@nb.njit('float64(float64[:])', nogil=True, fastmath=True)
def ententro(corr_eigvals):
    """
    A function for calculating the entanglement
    entropy from the eigenspectrum of the generalized
    one-body correlation matrix.
    """

    term1 = 0.5 * (1 + corr_eigvals)
    term2 = 0.5 * (1 - corr_eigvals)

    return -np.nansum(term1 * np.log(term1) + term2 * np.log(term2))
# define functions

# how big the subpartition is -> get the
# partition of the system


@nb.njit('uint64[:](uint64[:], float64)', nogil=True)
def get_subsystem(states, partition_fraction):
    """
    Get subsystem indices for restricting the
    subsystem. TO DO: allow for generalization,
    currently only homogenous bipartitions work.

    Parameters:

    states: ndarray, uint64, 1D
    Array of basis states

    dim_shape: ndarray, uint64, 1D
    Shape of the lattice.

    fraction: float
    Volume fraction of the partition

    Returns:
    indices: ndarray, uint64, 1D

    """
    return states[:int(partition_fraction * len(states))]


# select the many-body configuration;
# grand-canonical and canonical sampling
@nb.njit('boolean[:, :](boolean[:, :], uint64, uint64[:], uint32, float64, boolean)', nogil=True)
def _pick_mb_configuration(confs, state, states, seed, particle_filling, gc=True):
    """
    Pick indices of the single body states
    participating in the many-body energy eigenstate.

    Parameters:

    states: ndarray, uint64, 1D
    Array of basis states

    seed: seed for the random generator
    Here for reproducibility

    particle_filling: float, optional
    Fraction of states, defaults to 0.5 (half-filling)

    gc: boolean, optional
    Whether to sample the configuration from a
    grandcanonical (if True) or (micro)canonical
    ensemble (fixed number of particles)

    Returns:
    conf: ndarray, bool, 1D
    Boolean array used for slicing the eigenstate
    array.


    """
    # new way for instantiating
    # random generators with seeds
    np.random.seed(seed)
    filling = particle_filling

    if gc:
        # generate a boolean array for
        # slicing -> True/False for
        # each site with a probability
        # equal to the filling parameter
        # different random samples may thus
        # contain different numbers of
        # particles
        conf = np.random.choice(
            np.array([True, False]),
            states.shape[0], [filling, 1 - filling])
        for i in range(confs.shape[1]):

            confs[state][i] = conf[i]
    else:

        conf = np.zeros_like(states, dtype=bool_)
        conf[:int(filling * states.shape[0])] = True
        np.random.shuffle(conf)
        # now shuffle (this is inplace as opposed
        # to permute)
        for i in range(confs.shape[1]):
            confs[state][i] = conf[i]

        # conf = np.sort(np.random.choice(states, int(
        #    len(states) * filling), replace=False))

    return confs


# generate multiple configurations
@nb.njit('boolean[:, :](uint64[:], uint32, uint32, float64, boolean)', nogil=True)
def _generate_configurations_nb(states, seedmin, seedmax,
                                particle_filling, gc):
    """
    Generate an array of available configurations;
    NOTE: they need to be unique so this is why we have
    the generate_configurations() wrapper routine (numba
    does not support the numpy.unique() function yet.)

    """
    filling = particle_filling
    confs = np.zeros(
        (seedmax - seedmin, states.shape[0]), dtype=bool_)

    # i = 0
    j = seedmin

    for j in range(seedmin, seedmax):  # nb.prange(seedmin, seedmax):

        confs = _pick_mb_configuration(
            confs, j - seedmin, states, j, filling, gc)

        # i += 1
        # j += 1

    return confs
    # return np.unique(confs, axis=0)


def generate_configurations(states, seedmin, seedmax, particle_filling, gc):
    """
    A wrapper for the _generate_configurations_nb(...)
    routine which also makes sure the configurations
    are unique.

    """
    filling = particle_filling
    states = np.uint64(states)
    seedmin = np.uint32(seedmin)
    seedmax = np.uint32(seedmax)
    filling = np.float64(filling)

    confs = _generate_configurations_nb(states, seedmin, seedmax, filling, gc)

    return np.unique(confs, axis=0)


@nb.njit('float64(float64[:,:], uint64[:], uint64[:])', nogil=True, parallel=False)
def get_ententro_real(eigvecs, subsystem, mb_configuration):
    """
    Calculate the entaglement entropy for a given
    subsystem configuration and the given many-body
    configuration.

    Parameters:

    eigvecs: ndarray, 2D, np.complex128
        An array of single-particle eigenvectors
    subsystem: ndarray, 1D, np.uint64
        An array of indices designating the subpartition
        of the system
    mb_configuration: ndarray, 1D, np.uint64
        An array of indices (for the eigvecs array)
        designating which single-body states participate
        in the many-body configuration for which the
        entanglement entropy is calculated.

    Returns:

    np.float64: the entanglement entropy for a chosen
        bipartition and many-body configuration

    """
    #eigvecs = eigvecs.copy()
    # pick rows and columns according to our needs
    # this is needed for the calculation of the
    # generalized correlation matrix
    n_sites = subsystem.shape[0]
    n_states = mb_configuration.shape[0]

    # initialization of the corr. coefficients
    # array
    corr_coeffs = np.zeros((n_sites,
                            n_states), dtype=np.float64)

    corr_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    gen_corr_matrix = np.zeros_like(corr_matrix)
    corr_eigvals = np.zeros(n_sites, dtype=np.float64)

    for i in range(n_sites):  # nb.prange(n_sites):
        for j in range(n_states):
            #
            corr_coeffs[i][j] = eigvecs[subsystem[i]][mb_configuration[j]]

    corr_matrix = corr_coeffs @ (corr_coeffs.T)

    # subtract the diagonal (orthonormality) and multiply by 2 -> see
    # the definition above
    gen_corr_matrix = 2 * corr_matrix - \
        np.eye(corr_matrix.shape[0], dtype=np.float64)

    # we need the spectrum of the generalized
    # correlation matrix calculated above
    corr_eigvals = np.linalg.eigvalsh(gen_corr_matrix)
    # use the function from this module to
    # calculate the entanglement entropy
    return ententro(corr_eigvals)


@nb.njit('float64[:](float64[:], float64[:, :], uint64[:], boolean[:, :], float64)', nogil=True, parallel=False)
def entro_states(eentro, eigvecs, states, configurations,
                 partition_fraction=0.5):
    """
    Calculate the entanglement entropy for different
    many-body states (composed of some combinations
    of single-body states)


    """
    # number of configurations/mb states
    n_confs = eentro.shape[0]
    eentro = np.zeros(n_confs, dtype=np.float64)
    subsystem_indices = np.zeros(
        int(states.shape[0] * partition_fraction),
        dtype=np.uint64)

    subsystem_indices = get_subsystem(states, partition_fraction)

    for i in nb.prange(n_confs):  # nb.prange(n_confs):

        # mb_configuration = states[configurations[i]]

        eentro[i] = get_ententro_real(
            eigvecs.view(), subsystem_indices, states[configurations[i]])

    return eentro


def _test_fun_entro(t=-1., W=1, dim=3, L=10,
                    wseed=0, stateseedmax=10,
                    partition_fraction=0.5,
                    particle_filling=0.5,
                    gc=True):
    """
    This function is for testing (batteries included); for
    actual calculations, use the main_fun_entro() function
    """

    dim = np.uint64(dim)
    dim_shape = np.array([L for i in range(dim)], dtype=np.uint64)
    np.random.seed(seed=wseed)
    fields = np.random.uniform(-0.5 * W, 0.5 * W, dim_shape,)
    hamiltonian = ham(L, dim, t, fields, pbc=True, dtype=np.complex128)
    eigvals, eigvecs = hamiltonian.eigsystem(complex=False)
    states = hamiltonian.states
    configurations = generate_configurations(states, np.uint64(0),
                                             np.uint64(stateseedmax),
                                             np.float64(particle_filling),
                                             gc)

    eentro = np.zeros(configurations.shape[0], dtype=np.float64)
    return eigvecs, entro_states(eentro, eigvecs.view(), states,
                                 configurations, partition_fraction)


def main_fun_entro(eigvecs, states, stateseedmax,
                   partition_fraction, particle_filling, gc):

    #eigvals, eigvecs = hamiltonian.eigsystem(complex=False)
    #states = hamiltonian.states
    configurations = generate_configurations(np.uint64(states),
                                             np.uint64(0),
                                             np.uint64(stateseedmax),
                                             np.float64(particle_filling),
                                             gc)
    eentro = np.zeros(configurations.shape[0], dtype=np.float64)
    return entro_states(eentro, np.float64(eigvecs),
                        states, configurations,
                        partition_fraction)


def main_fun_entro_meminfo(eigvecs, states, stateseedmin, stateseedmax,
                           partition_fraction, particle_filling, gc, p):

    #eigvals, eigvecs = hamiltonian.eigsystem(complex=False)
    #states = hamiltonian.states
    print(f'Memory used #1: {p.memory_info().rss/1e06} MB.')
    configurations = generate_configurations(np.uint64(states),
                                             np.uint64(stateseedmin),
                                             np.uint64(stateseedmax),
                                             np.float64(particle_filling),
                                             gc)
    eentro = np.zeros(configurations.shape[0], dtype=np.float64)
    print(f'Memory used #2: {p.memory_info().rss/1e06} MB.')
    eentro = entro_states(eentro, np.float64(eigvecs),
                          states, configurations,
                          partition_fraction)

    print(f'Memory used #3: {p.memory_info().rss/1e06} MB.')
    return eentro
