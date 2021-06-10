import numpy as np
from itertools import permutations

from anderson.operators import get_idx


def _nn_hoppings(dim):
    # return nearest neighbour
    # hoppings - only in the positive
    # directions
    phop = np.eye(dim, dtype=np.uint64)

    return phop


def _snn_hoppings(dim):
    """
    Find all allowed second
    nearest neighbour hoppings in
    the Anderson model (to be specific,
    on a simple cubic lattice)
    Note: only positive hoppings

    """
    if dim == 1:
        allowed_hoppings = np.array([[2], ], dtype=np.uint64)
    else:
        allowed_hoppings = np.array([[1, 1], ], dtype=np.uint64)

        if dim > 2:
            allowed_hoppings_ = np.zeros((1, dim), dtype=np.uint64)
            allowed_hoppings_[:, :2] = allowed_hoppings
            allowed_hoppings = []

            for hopping in allowed_hoppings_:
                allowed_hoppings += list(set(permutations(hopping)))

            allowed_hoppings = np.array(allowed_hoppings, dtype=np.uint64)

    return allowed_hoppings


def _format_hop_strings(hopping, hop_type):

    if hop_type == 0:
        hop_str = 'eigvec_components_central_site'
    else:
        if hop_type == 1:
            hop_str = 'eigvec_components_nn_hop_'
        if hop_type == 2:
            hop_str = 'eigvec_components_snn_hop_'

        hop_str += '_'.join(str(hop) for hop in hopping)

    return hop_str


def _prepare_noninteracting(model, nconv, qulist=[0.1, 0.5, 2.0]):
    """
    Prepare the arrays storing the results
    for the noninteracting Anderson model shift
    and invert code.

    """

    # ------------------------------------------------------
    #
    #   Initialize lists and dictionaries
    #
    # ------------------------------------------------------

    idx_dict = {}
    q_dict = {}
    ni_keys = ['Density_diag_matelts_partial']
    dtypes = [np.float64]

    # ------------------------------------------------------
    #
    #   Prepare hoppings, get indices of the states
    #
    # ------------------------------------------------------
    # prepare hoppings and get indices of the matrix element
    # matelt coordinates & idx
    _matelt_coordinates = np.array([int(0.5 * model.L)
                                    for i in range(model.dim)],
                                   dtype=np.uint64)
    _dimension = np.array(
        [model.L for i in range(model.dim)], dtype=np.uint64)
    # what is the index of the site in the occupational basis
    _idx = get_idx(_matelt_coordinates, _dimension)

    nn_hops = _nn_hoppings(model.dim)
    snn_hops = _snn_hoppings(model.dim)

    # ------------------------------------------------------
    #
    #  Format descriptor strings
    #
    # ------------------------------------------------------

    central_string = _format_hop_strings([], 0)
    ni_keys.append(central_string)
    dtypes.append(np.complex128)
    idx_dict[ni_keys[0]] = _idx
    idx_dict[central_string] = _idx

    # format result strings and
    # dtypes for hoppings
    for i, hops in enumerate([nn_hops, snn_hops]):
        for hop in hops:

            hop_string = _format_hop_strings(hop, i + 1)
            ni_keys.append(hop_string)
            dtypes.append(np.complex128)
            idx_dict[hop_string] = get_idx(_matelt_coordinates + hop,
                                           _dimension)

    for q_ in qulist:
        ipr_key = f'Ipr_q_{q_:2f}_partial'
        ni_keys.append(ipr_key)
        dtypes.append(np.float64)
        q_dict[ipr_key] = q_

    ni_results = {}
    for key, dtype in zip(ni_keys, dtypes):
        ni_results[key] = np.zeros(nconv, dtype=dtype)

    # print(idx_dict)
    # print(ni_results)
    return ni_results, idx_dict, q_dict


def _analyse_noninteracting(eigpair_idx, eigvec,
                            results_dict, idx_dict, q_dict):
    """
    Prepare all the analysis for
    the noninteracting case
    that has to be done on
    an extracted eigenvector

    # we should extract:
    eigenvalues
    ipr (3 different q-values)
    3 different types of matrix
    elements (density, hopping,
    diagonal hopping)

    """
    # keys for storing results

    for key, value in idx_dict.items():
        # print(key)

        if key == 'Density_diag_matelts_partial':
            #print('density diag partial')
            results_dict[key][eigpair_idx] = np.abs(
                eigvec[value])**2
        else:
            results_dict[key][eigpair_idx] = eigvec[value]

    for key, value in q_dict.items():

        results_dict[key][eigpair_idx] = np.sum(np.abs(eigvec)**(2 * value))

    return results_dict
