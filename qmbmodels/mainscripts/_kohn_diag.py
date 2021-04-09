#!/usr/bin/env python

"""


"""
import numpy as np

# We separate the following cases:
# A) INTERACTING
# B) NONTINTERACTING
#
#   NONINTERACTING CASE FURTHER DIVIDES
#   INTO COMPLEX AND REAL CASES


def _return_cases(interacting, complex_):
    # anderson and real -> Thouless conductivity
    # calculation
    case_1 = ((not interacting) and (not complex_))
    # Kohn conductivity in the Anderson case
    case_2 = ((not interacting) and complex_)
    # only the general Kohn conductivity, for Thouless
    # case, set the phase to pi
    case_3 = (interacting and (not complex_))

    case_4 = (interacting and complex_)

    return case_1, case_2, case_3, case_4


def _prep_unperturbed(argsDict, interacting, complex_,
                      keys=['phase_bc', 'J_phase']):
    """
    Prepare the unperturbed matrix parameters for calculations
    """
    # separate different cases
    # because we need to prepare the unperturbed
    # hamiltonians differently

    case_1, case_2, case_3, case_4 = _return_cases(interacting,
                                                   complex_)

    argsDict_ = argsDict.copy()
    changed_entries = []

    if case_1:
        print('Preparation message: noninteracting, real case')
    elif case_2:
        print('Preparation message: noninteracting, complex case')
    elif (case_3 or case_4):

        if case_3:
            print('Preparation message: interacting, real case')

        if case_4:
            print('Preparation message: interacting, complex case')

        for key_ in keys:
            changed_entries.append(argsDict_[key_])
            argsDict_[key_] = 0.

    if ((case_1) or (case_3)):

        savename_1 = 'Spectrum_apbc'
        savename_2 = 'Spectrum_differences'
    elif ((case_2) or (case_4)):
        if case_2:
            phase_factor = argsDict['phase_bc']
        if case_4:
            # in the main code,
            # we explicitly require that one is nonzero;
            # if both are nonzero, this turns to case_3
            phase_factor = [argsDict[key_] for
                            key_ in keys if argsDict[key_] != 0.][0]

        savename_1 = f'Spectrum_phase_factor_{phase_factor:.8f}'
        savename_2 = f'Spectrum_differences_phase_factor_{phase_factor:.8f}'

    return argsDict_, changed_entries, savename_1, savename_2


def _prep_perturbed(argsDict, interacting, complex_,
                    changed_entries,
                    keys=['phase_bc', 'J_phase']):
    """
    Prepare the perturbed matrix parameters for calculations
    """
    # separate different cases
    # because we need to prepare the unperturbed
    # hamiltonians differently

    case_1, case_2, case_3, case_4 = _return_cases(interacting,
                                                   complex_)

    argsDict_ = argsDict.copy()


    if case_1:
        argsDict_['pbc'] = np.array(
            [1 for i in range(argsDict['dim'])], dtype=np.int32)
        argsDict_['pbc'][-1] = -1
    elif case_2:
        argsDict_['pbc'] = np.array(
            [1 for i in range(argsDict['dim'])], dtype=np.complex128)
        argsDict_['pbc'][-1] = argsDict_['mod_bc'] * \
            np.exp(1j * argsDict_['phase_bc'])
    elif (case_3 or case_4):

        for dict_ in [argsDict, argsDict_]:
            for i, key_ in enumerate(keys):

                dict_[key_] = changed_entries[i]

            if case_3:
                dict_['phase_bc'] = np.pi

    return argsDict, argsDict_

def kohn_diag(mod, argsDict, interacting, complex_=True):

    argsDict_unp, changed_entries, savename_1, savename_2 = _prep_unperturbed(
        argsDict,
        interacting,
        complex_)

    model, fields = mod.construct_hamiltonian(argsDict_unp, parallel=False,
                                              mpisize=1)
    # get the instance of the appropriate hamiltonian
    # class and the diagonal random fields used
    print('argsdict: {}'.format(argsDict_unp))
    model, fields = mod.construct_hamiltonian(
        argsDict_unp, parallel=False, mpisize=1)

    print('Starting diagonalization for the unperturbed case...')
    # save some time and memory, the matrices should be
    # real at this point
    eigvals_unp = model.eigvals(complex=complex_)

    # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
    print('Diagonalization for the unperturbed case finished!')

    # prepare the perturbed case

    argsDict_unp, argsDict_pert = _prep_perturbed(argsDict_unp,
                                                  interacting,
                                                  complex_,
                                                  changed_entries)

    print('argsdict_perturbedc: {}'.format(argsDict_pert))
    model, fields = mod.construct_hamiltonian(
        argsDict_pert, parallel=False, mpisize=1)
    print('Starting diagonalization for the perturbed case...')
    eigvals_pert = model.eigvals(complex=complex_)

    # eigvals, eigvecs = model.eigsystem(complex=False, turbo=True)
    print('Diagonalization for the perturbed case finished!')

    spectrum_differences = eigvals_unp - eigvals_pert
    print('Displaying differences between spectra:')
    print(spectrum_differences)

    # prepare for saving

    eigvals_dict = {'Eigenvalues': eigvals_unp,
                    savename_1: eigvals_pert,
                    savename_2: spectrum_differences,
                    **fields}

    return eigvals_dict, argsDict_unp
