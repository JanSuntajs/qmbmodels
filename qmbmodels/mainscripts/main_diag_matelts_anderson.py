#!/usr/bin/env python
"""
This module is specifically intended for the analysis of
Anderson model and the matrix elements of observables
since those have a special and rather simple form
in the Anderson model. This code also calculates
the eigenvectors so it should be used with care!

"""
from itertools import permutations
import numpy as np

from anderson.operators import get_idx
from qmbmodels.utils import set_mkl_lib
from qmbmodels.utils.cmd_parser_tools import arg_parser, arg_parser_general
from qmbmodels.models.prepare_model import get_module_info
from qmbmodels.utils.filesaver import savefile
from qmbmodels.models.prepare_model import select_model

save_metadata = True

def _reshape_spectrum(array, store_nener):

    nener_orig = array.shape[1]
    if store_nener == -1:
        return array, 0, nener_orig
    else:
        remain = nener_orig - store_nener
        emin = int(0.5 * remain)
        emax = emin + store_nener
        return array[:, emin:emax], emin, emax


def nn_hoppings(dim):
    # return nearest neighbour
    # hoppings
    phop = np.eye(dim, dtype=np.int64)
    mhop = -np.eye(dim, dtype=np.int64)

    return np.vstack((phop, mhop))


def snn_hoppings(dim):
    """
    Find all allowed second
    nearest neighbour hoppings in
    the Anderson model (to be specific,
    on a simple cubic lattice)

    """
    if dim == 1:
        allowed_hoppings = np.array([[2], [-2]], dtype=np.int64)
    else:
        allowed_hoppings = np.array([[1, 1], [-1, -1],
                                     [-1, 1], [1, -1]], dtype=np.int64)

        if dim > 2:
            allowed_hoppings_ = np.zeros((3, dim), dtype=np.int64)
            allowed_hoppings_[:, :2] = allowed_hoppings[:-1, :]
            allowed_hoppings = []

            for hopping in allowed_hoppings_:
                allowed_hoppings += list(set(permutations(hopping)))

            allowed_hoppings = np.array(allowed_hoppings)

    return allowed_hoppings


# how many eigenpairs to use in the matrix elements calculations
_matelts_parse_dict = {'matelts_nener': [int, -1]}

if __name__ == '__main__':

    mateltsDict, matelts_extra = arg_parser_general(_matelts_parse_dict)
    store_nener = mateltsDict['matelts_nener']
    print(mateltsDict)

    (mod, model_name, argsDict, seedDict, syspar_keys,
     modpar_keys, savepath, syspar, modpar, min_seed,
     max_seed) = get_module_info()

    save_metadata = True

    nn_hops = nn_hoppings(argsDict['dim'])
    snn_hops = snn_hoppings(argsDict['dim'])

    for seed in range(min_seed, max_seed + 1):
        print('Using seed: {}'.format(seed))
        argsDict['seed'] = seed
        # get the instance of the appropriate hamiltonian
        # class and the diagonal random fields used
        model, fields = mod.construct_hamiltonian(
            argsDict, parallel=False, mpisize=1)

        print('Starting diagonalization ...')
        eigvals, eigvectors = model.eigsystem(complex=True)

        dimension = np.array(
            [model.L for i in range(model.dim)], dtype=np.uint64)

        # how many energies are
        # there in the spectrum (Hilbert
        # space dimension)
        eigvectors, emin, emax = _reshape_spectrum(eigvectors,
                                                   store_nener)
        # store energies from the centre of the spectrum here
        print(f'e_min idx: {emin}')
        print(f'e_max idx: {emax}')

        # matrix element index - we choose the one in the middle of
        # the chain to avoid possible finite-size effects - the density
        # operator acts on a site in the middle of the chain
        matelt_coordinates = np.array([0.5 * model.L
                                       for i in range(model.dim)],
                                      dtype=np.uint64)

        # -----------------------------------------------------
        #
        # LOCAL DENSITY MATRIX ELEMENTS CALCULATION
        #
        # ONLY DIAGONALS FOR NOW
        #
        #
        # -----------------------------------------------------
        # get the matrix element index in the Hamiltonian representation
        # in the coordinate representation, lattice sites equal the basis
        # states; for easier Hamiltonian representation, we represent
        # the Hamiltonian lattice as a 1D string. We need to get the
        # basis state index from the provided operator coordinates, which
        # is done by the internal get_idx(...) function
        dens_idx = get_idx(matelt_coordinates, dimension)

        # take only the diagonals
        dens_coeffs = np.abs(eigvectors[dens_idx, :])**2

        dens_dict = {f'central_density_emin_{emin}_emax_{emax}': dens_coeffs}
        # -----------------------------------------------------
        #
        # NEAREST NEIGHBOUR HOPPINGS
        #
        # ONLY DIAGONALS FOR NOW -> just take the dot
        # product, not the tensor product and get only
        # the diagonal matrix elements
        #
        # -----------------------------------------------------

        hop_dicts = [{}, {}]
        hop_types = [nn_hops, snn_hops]
        hop_strings = ['nn_hop', 'snn_hop']

        for j, hop_type in enumerate(hop_types):
            for hopping in hop_type:
                print(f'{hop_strings[j]}: {hopping}.')
                hop_idx = get_idx(
                    np.uint64(matelt_coordinates + hopping),
                    dimension)
                hop_coeffs = np.conj(
                    eigvectors[dens_idx, :]) * eigvectors[hop_idx, :]
                hop_desc = '_'.join(str(hop) for hop in hopping)
                hop_dicts[j][(f'{hop_strings[j]}_emin_{emin}_'
                              f'emax_{emax}_hop_{hop_desc}')] = hop_coeffs

        print('Diagonalization finished!')

        #print('Displaying eigvals')
        # print(eigvals)
        print('hopdicts[0]')
        print(hop_dicts[0])
        # ----------------------------------------------------------------------
        # save the files
        eigvals_dict = {'Eigenvalues': eigvals,
                        **fields,
                        **dens_dict,
                        **hop_dicts[0],
                        **hop_dicts[1]}

        savefile(eigvals_dict, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'full',
                 save_metadata, save_type='npz')

        save_metadata = False
