"""
This model implements the toy model
for the simulation of the quantum
sun model in the simplest nontrivial form,
e. g., as a sequence of GOE block-diagonal
matrices

The model, intended as a simulation of
the quantum avalanche process, initially
consists of a random grain of size N and
L localized spins outside the grain. We then
couple the spins to the grain in an exponential
manner using a coupling parameter \alpha. The
full Hilbert space dimension thus equals
2**(L + N).

In the zeroth order, the Hamiltonian
is a block diagonal matrix of 2**L blocks of size
2**N, where each block is an independent GOE matrix.

In the first order in coupling, we add the off-diagonal
matrix blocks coupling the neighbouring diagonal sectors,
hence the matrix structure of the first interaction term
is 2**(L - 1) GOE blocks of size 2**(N + 1), multiplied
by the coupling of \alpha^1. We proceed in such a manner
until all of the localized spins are included in the grain
and hence the coupling term is a single GOE matrix of
size 2**(L + N).

The full Hamiltonian is thus written as:

H = H_0 + \alpha H_1 + \alpha^2 H_2 + ... + \alpha^L H_L

IMPORTANT:

We ensure that the Hilbert-Schmidt (HS) norm of the Hamiltonian term
on each step equals unity. The HS norm  of an operator
A is defined as follows:

|| A ||_HS = (1/D) \Tr{AA*},
where \Tr is the trace and D is the Hilbert space dimension. To fulfill
the requirement of unit HS norm in the actual simulations, we let the 
GOE submatrices of the Hamiltonian terms have unit variance; then,
we rescale the matrix elements by 2**(-(N + i)/2) on i-th step
of the procedure.


"""


import numpy as np

from ham1d.models.usrdef import hamiltonian as ushm

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'L_b'] + comm_syspar_keys
modpar_keys = ['alpha'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def rmat(N, scale=np.sqrt(2)):
    """
    Construct a random Gaussian matrix
    with a unit Hilbert-Schmidt norm

    Parameters:

    N: int,
    determines the size of the rmat
    according to dim = 2**N

    scale: float, optional
    Parameter determining the variance of the
    Gaussian matrix a_ij; defaults to \sqrt(2)
    """

    amat = np.random.normal(size=(2**N, 2**N), loc=0., scale=scale)

    return 0.5 * (amat + amat.T)


def _gen_step(N, L, alpha):
    """
    Parameters:

    N: int
    Initial grain size

    L: int
    Number of localized spins.

    alpha: float
    Coupling strength

    Returns:

    mat: ndarray, 2D
    2D ndarray, a full Hamiltonian matrix

    matlist: list
    A list of 2D ndarrays, for normalized
    operators on each step (starting with
    the zeroth step, alpha is not included here.)

    """

    Lful = N + L
    mat = np.zeros((2**Lful, 2**Lful))

    # list of matrices on each step,
    # without alpha,
    # each entry in matlist is normalized
    # such that it's H-S norm
    # 1/2**L ||A||_F = 1
    matlist = []
    j = 0
    for j, N_ in enumerate(range(N, Lful + 1)):
        # print(N_)

        ldim = 2**N_
        temp = np.zeros_like(mat)
        for i in range(2**(Lful - N_)):

            temp_ = rmat(N_)
            # proper normalization according to the
            # unit HS norm
            temp_ *= 1./np.sqrt(2**(N_))
            temp[i*ldim:(i+1)*ldim, i*ldim:(i+1)*ldim] = temp_

        mat += temp * alpha**j

        matlist += [temp]
        j += 1

    return mat, matlist


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0,
                          dtype=np.float64):
    """


    """

    L = argsdict['L']
    L_b = argsdict['L_b']

    L_loc = int(L - L_b)

    alpha = argsdict['alpha']

    if L_b < 1:
        raise ValueError(('Bath length L_b should '
                          'be greater than 0!'))

    # no pbc by design hereo
    if argsdict['pbc']:

        raise ValueError('No pbc allowed in this model!')

    ham_type = argsdict['ham_type']

    if ham_type != 'usrdef':

        raise ValueError('Only usrdef ham_type allowed here!')

    ham = ushm

    # get the hamiltonian matrix and the list of
    # terms
    seed = argsdict['seed']
    mat, *_ = _gen_step(L_b, L_loc, alpha)
    # for usrdef hamiltonians:

    params_ = {
        'L': L,
        'L_b': L_b,
        'alpha': alpha,
        'seed': seed,

    }

    hamiltonian = ham(basis=[], mat=mat, params=params_,
                      parallel=False, mpirank=0, mpisize=0,
                      dtype=dtype)

    return hamiltonian, {}
