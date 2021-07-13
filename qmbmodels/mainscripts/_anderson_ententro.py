"""
To avoid code repetition, we implement
the routines needed in the calculation
of the many-body entanglement entropy
for the (any-dimensional) Anderson model
in a separate module displayed here.


"""

import numpy as np
import numba as nb
from scipy.special import comb
from scipy.linalg import eigh
from itertools import permutations

from anderson.model import hamiltonian as ham
from anderson.operators import get_idx, get_coordinates


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


@nb.njit('uint64[:](uint64[:], float64)', nogil=True)
def get_subsystem(states, fraction):
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
    Volume fraction of the partion

    Returns:
    indices: ndarray, uint64, 1D

    """
    return states[:int(fraction * len(states))]


@nb.njit('uint64[:](uint64[:], uint32, float64)', nogil=True)
def _pick_mb_configuration(states, seed, filling):
    """
    Pick indices of the single body states
    participating in the many-body energy eigenstate.

    Parameters:

    states: ndarray, uint64, 1D
    Array of basis states

    seed: seed for the random generator
    Here for reproducibility

    filling: float, optional
    Fraction of states, defaults to 0.5 (half-filling)

    Returns:
    conf: ndarray, uint64, 1D    


    """

    np.random.seed(seed)
    conf = np.sort(np.random.choice(states, int(
        len(states) * filling), replace=False))

    return conf


@nb.njit('uint64[:, :](uint64[:], uint32, uint32, float64)', nogil=True)
def _generate_configurations_nb(states, seedmin, seedmax, filling):
    """
    Generate an array of available configurations;
    NOTE: they need to be unique so this is why we have
    the generate_configurations() wrapper routine (numba
    does not support the numpy.unique() function yet.)

    """
    confs = np.zeros(
        (seedmax - seedmin, int(filling * states.shape[0])), dtype=np.uint64)

    #i = 0
    j = seedmin

    for i, j in enumerate(range(seedmin, seedmax)):

        _conf = _pick_mb_configuration(states, i, filling)
        confs[i] = _conf
        #i += 1
        #j += 1

    return confs
    # return np.unique(confs, axis=0)


def generate_configurations(states, seedmin, seedmax, filling):
    """
    A wrapper for the _generate_configurations_nb(...)
    routine which also makes sure the configurations
    are unique.

    """
    states = np.uint64(states)
    seedmin = np.uint32(seedmin)
    seedmax = np.uint32(seedmax)
    filling = np.float64(filling)

    confs = _generate_configurations_nb(states, seedmin, seedmax, filling)

    return np.unique(confs, axis=0)


@nb.njit('float64(float64[:,:], uint64[:], uint64[:])', nogil=True, parallel=True)
def get_ententro(eigvecs, subsystem, mb_configuration):
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
    eigvecs = eigvecs.copy()
    # pick rows and columns according to our needs
    # this is needed for the calculation of the
    # generalized correlation matrix
    n_sites = subsystem.shape[0]
    n_states = mb_configuration.shape[0]

    # initialization of the corr. coefficients
    # array
    corr_coeffs = np.zeros((n_sites,
                            n_states), dtype=np.complex128)

    for i in range(n_sites):
        for j in range(n_states):
            #
            corr_coeffs[i][j] = eigvecs[subsystem[i]][mb_configuration[j]]

    corr_matrix = corr_coeffs @ (np.conjugate(corr_coeffs.T))

    # subtract the diagonal (orthonormality) and multiply by 2 -> see
    # the definition above
    gen_corr_matrix = 2 * corr_matrix - \
        np.eye(corr_matrix.shape[0], dtype=np.complex128)

    # we need the spectrum of the generalized
    # correlation matrix calculated above
    corr_eigvals = np.linalg.eigvalsh(gen_corr_matrix)
    # use the function from this module to
    # calculate the entanglement entropy
    return ententro(corr_eigvals)


@nb.njit('float64[:](float64[:, :], uint64[:], uint64[:, :], float64)', nogil=True, parallel=True)
def entro_states(eigvecs, states, configurations, filling=0.5):
    """
    Calculate the entanglement entropy for different
    many-body states (composed of some combinations
    of single-body states)


    """
    # number of configurations/mb states
    n_confs = configurations.shape[0]

    subsystem_indices = get_subsystem(states, filling)
    eentro = np.zeros(n_confs, dtype=np.float64)
    for i in nb.prange(n_confs):

        mb_configuration = configurations[i]

        eentro[i] = get_ententro(eigvecs, subsystem_indices, mb_configuration)

    return eentro


def _test_fun_entro(t=-1., W=1, dim=3, L=10,
                    wseed=0, stateseedmax=10, filling=0.5):
    """
    This function is for testing (batteries included); for
    actual calculations, use the main_fun_entro() function
    """

    dim = np.uint64(dim)
    dim_shape = np.array([L for i in range(dim)], dtype=np.uint64)
    np.random.seed(seed=wseed)
    fields = np.random.uniform(-0.5 * W, 0.5 * W, dim_shape,)
    hamiltonian = ham(L, dim, t, fields, pbc=True, dtype=np.complex128)
    eigvals, eigvecs = hamiltonian.eigsystem(complex=True)
    states = hamiltonian.states
    configurations = generate_configurations(states, np.uint64(0),
                                             np.uint64(stateseedmax),
                                             np.float64(filling))
    return entro_states(eigvecs, states, configurations, filling)


def main_fun_entro(hamiltonian, stateseedmax, filling):

    eigvals, eigvecs = hamiltonian.eigsystem(complex=True)
    states = hamiltonian.states
    configurations = generate_configurations(states, np.uint64(0),
                                             np.uint64(stateseedmax), np.float64(filling))

    return eigvals, entro_states(np.float64(eigvecs), states, configurations,
                                 filling)
