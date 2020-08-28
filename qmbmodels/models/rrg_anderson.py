"""
This module contains the tools for construction
of the 1D Anderson random regular graph model. For
more details on the model, see for instance:

https://arxiv.org/abs/1903.07877
(K. S. Tikhonov, A. D. Mirlin, Critical behavior
at the localization transition on random regular
graphs)

IMPORTANT: IN THE ANDERSON PICTURE, DISORDER IS USUALLY
SAMPLED FROM AN INTERVAL -0.5*dW, 0.5*dW -> this is accounted
for in our code here.

Attributes:
-----------

syspar_keys: list
    System parameter descriptors.
    L: int, system size exponent; the number of nodes
    equals N=2**L
    K: local connectivity indicating the number of adjacent
    (directly connected) nodes for each node.
    The rest: see the docstring for
    the _common_keys module.

modpar_keys: list
    Module parameters for the
    Heisenberg model:
    't',  'W', 'dW': float
    - nearest neighbour exchange for states adjacent on
    a graph, not in the actual coordinate space.
    - center of the disorder distribution and width of the
    disorder distribution, respectively
"""


import numpy as np
from networkx.generators.random_graphs import random_regular_graph


from ham1d.models.free1d import hamiltonian as ham

from .disorder import get_disorder_dist
from ._common_keys import comm_modpar_keys, comm_syspar_keys

syspar_keys = ['L', 'K'] + comm_syspar_keys
modpar_keys = ['t', 'W', 'dW'] + comm_modpar_keys

_modpar_keys = [key for key in modpar_keys if '_seed' not in key]
_modpar_keys.append('seed')


def construct_hamiltonian(argsdict, parallel=False, mpirank=0, mpisize=0):
    """
    Constructs the Anderson random regular graph (RRG) model in
    one dimension. For
    more details on the model, see for instance:

    https://arxiv.org/abs/1903.07877
    (K. S. Tikhonov, A. D. Mirlin, Critical behavior
    at the localization transition on random regular
    graphs)


    Parameters:
    -----------

    argsdict: dict
        A dictionary containing the information needed to construct
        the Hamiltonian. Should have the following keys:
        'L': int, exponent in the system size/number of nodes
        of a random regular graph, N = 2**L, where N is the
        number of nodes.
        'K': int, local connectivity specifying to how many
        adjacent points a node connects to.
        'disorder': str, specifies the disordered potential used.
        'ham_type': str, which Hamiltonian type is used. Only
                    'free1d' is supported, as this Hamiltonian
                    is intended a free fermionic one.
        't', 'W', 'dW': float
            Model parameters for the RRG 1D Anderson model: nearest
            neighbour echange couplings, center of
            the disorder distribution and width of the disorder
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
        ham1d package

    dict:
        A dictionary containing the potential disorder
        under the key 'Hamiltonian_random_disorder'
        and a flattened array of random links between nodes.

    Raises:
    -------

    ValueError if argsdict['ham_type'] != 'free1d'

    """

    L = argsdict['L']
    # the number of nodes of a random regular graph
    N = 2**L
    # local connectivity
    K = argsdict['K']

    # generate the random regular graph edges
    graph_edges = np.array(random_regular_graph(
        K, N, seed=argsdict['seed']).edges())
    disorder = argsdict['disorder']

    ham_type = argsdict['ham_type']

    if ham_type != 'free1d':

        err_message = ('ham_type {} not allowed for the'
                       'RRG Anderson model! Only free1d '
                       'case is valid!').format(ham_type)
        raise ValueError(err_message)

    hops = [['+-', [[argsdict['t'], *coup] for coup in graph_edges]],
            ['+-', [[argsdict['t'], *coup[::-1]] for coup in graph_edges]],
            ]
    fields = get_disorder_dist(N, disorder, argsdict['W'],
                               0.5 * argsdict['dW'], argsdict['seed'])
    rnd = ['n', [[field, i] for i, field in enumerate(fields)]]

    static_list = [*hops, rnd]

    hamiltonian = ham(N, static_list, [], parallel=parallel,
                      mpirank=mpirank, mpisize=mpisize)

    return hamiltonian, {'Hamiltonian_random_disorder': fields,
                         'Hamiltonian_random_graph_edges': graph_edges.flatten()}
