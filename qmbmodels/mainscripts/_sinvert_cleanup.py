import numpy as np
from scipy import linalg as LA
from itertools import permutations

from ham1d.entropy.rdm import build_rdm
from ham1d.entropy.ententro import Entangled
from anderson.operators import get_idx
from qmbmodels.utils.filesaver import savefile
from qmbmodels.utils.cmd_parser_tools import arg_parser_general


def _nn_hoppings(dim):
    # return nearest neighbour
    # hoppings - only in the positive
    # directions
    phop = np.eye(dim, dtype=np.int64)

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
        allowed_hoppings = np.array([[2], ], dtype=np.int64)
    else:
        allowed_hoppings = np.array([[1, 1], ], dtype=np.int64)

        if dim > 2:
            allowed_hoppings_ = np.zeros((1, dim), dtype=np.int64)
            allowed_hoppings_[:, :2] = allowed_hoppings
            allowed_hoppings = []

            for hopping in allowed_hoppings_:
                allowed_hoppings += list(set(permutations(hopping)))

            allowed_hoppings = np.array(allowed_hoppings)

    return allowed_hoppings


def _format_hop_strings(hopping, hop_type):

    if hop_type == 0:
        hop_str = 'eigvec_components_central_site'
    else:
        if hop_type == 1:
            hop_str = 'eigvec_components_nn_hop'
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
    ni_keys = []
    dtypes = []

    # ------------------------------------------------------
    #
    #   Prepare hoppings, get indices of the states
    #
    # ------------------------------------------------------
    # prepare hoppings and get indices of the matrix element
    # matelt coordinates & idx
    _matelt_coordinates = np.array([0.5 * model.L
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
    idx_dict[central_string] = _idx

    # format result strings and
    # dtypes for hoppings
    for i, hops in enumerate([nn_hops, snn_hops]):
        for hop in hops:
            hop_string = _format_hop_strings(hop, i)
            ni_keys.append(hop_string)
            dtypes.append(np.complex128)
            idx_dict[hop_string] = get_idx(_matelt_coordinates + hop,
                                           _dimension)

    for q_ in qulist:
        ipr_key = f'Ipr_q_{q:2f}_partial'
        ni_keys.append(ipr_key)
        dtypes.append(np.float64)
        q_dict[ipr_key] = q_

    ni_results = {}
    for key, dtype in zip(ni_keys, dtypes):
        ni_results[key] = np.zeros(nconv, dtype=dtype)

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

        results_dict[key][eigpair_idx] = eigvec[val]

    for key, value in q_dict.items():

        results_dict[key] = np.sum(np.abs(eigvec)**(2 * value))

    return results_dict


def _sinvert_save(eigvals, fields, results_dict, metadata,
                  save_space, savepath, syspar, modpar,
                  argsDict, syspar_keys, modpar_keys):

    sortargs = np.argsort(eigvals)
    eigvals = np.array(eigvals)[sortargs]


def sinvert_collect_and_save(model,
                             matrix, E_si, ener0, ener1,
                             mpirank, PETSc,
                             savepath, syspar, modpar,
                             argsDict, syspar_keys,
                             modpar_keys,
                             many_body,
                             save_space):

    nconv = E_si.getConverged()
    print('nconv is:')
    print(nconv)

    """
    Extract, collect and store
    the shift and invert results
    """
    # ------------------------------------------------------------
    #
    #       EXTRACTION OF EIGENVECTORS AND EIGENVALUES
    #       ENTANGLEMENT ENTROPY CALCULATION
    #       IPR CALCULATION FOR NON-MANY BODY MODELS
    #       IPR - INVERSE PARTICIPATION RATIO
    #
    # ------------------------------------------------------------

    # for anderson case: extract the diagonal matrix elements
    # of the density operator for the central site
    # initialize things first

    # store to a numpy array
    if nconv > 0:

        # for finding the eigenvectors

        vr, tmp = matrix.getVecs()
        vi, tmp = matrix.getVecs()

        # if mpirank == 0:

        eigvals = np.zeros(nconv, dtype=np.complex128)
        if not many_body:

            # initialize results, indices for
            # matelts, q_values for ipr
            results_dict, idx_dict, q_dict = _prepare_noninteracting(
                model, nconv)

        for i in range(0, nconv):

            # Get an eigenpair where eigval
            # is the eigenvalue, i is the
            # eigenvalue index and
            # vr, vi are the real and complex
            # component of the eigenvector.
            # FOR NOW: currently, the matrices
            # we are considering are in fact
            # real and symmetric, so the
            # the eigenvectors are purely real,
            # hence only the vr part matters to
            # us. This might change in the future!
            eigval = E_si.getEigenpair(i, vr, vi)

            # create a scatter context
            # We gather the global vector from
            # all the matrix processes on the
            # zeroth process.
            ctx = PETSc.Scatter()

            # tozero is a scatter context
            # intended to collect all the
            # global vector components on the
            # zeroth process.
            # zvec is the PETSc.Vec() type
            # that actually stores the vector
            # values. In case we wanted to
            # obtain the imaginary components,
            # we would have to repeat the
            # procedure for the imaginary unless
            # PETSc was compiled in the complex
            # arithmetic.
            tozero, zvec = ctx.toZero(vr)
            # itozero, zveci = ctxi.toZero(vi)

            # fill the vector with values
            # syntax:
            # scatterBegin(self, Vec vec_from,
            #              Vec vec_to, addv=None,
            #              mode=None)
            # And the same interface for scatterEnd
            tozero.scatterBegin(vr, zvec, 1, 0)
            tozero.scatterEnd(vr, zvec, 1, 0)

            # calculate entanglement entropy
            # do this on the first process where
            # the vector is composed from all the
            # processes
            if mpirank == 0:
                eigvals[i] = eigval
                eigvec = np.array(zvec)
                # eigvecs.append(eigvec)

                results_dict = _analyse_noninteracting(i, eigvec,
                                                       idx_dict, q_dict)

            # destroy the scatter context before the new
            # loop iteration
            ctx.destroy()
            tozero.destroy()
            zvec.destroy()

            # save part - call the internal save function
            if mpirank == 0:


def sinvert_setup(model, mpirank, mpisize, comm,
                  PETSc, SLEPc, savepath, syspar,
                  modpar, argsDict, syspar_keys,
                  modpar_keys):
    # prepare for parallel PETSc assembly
    nnz = (model._d_nnz, model._o_nnz)

    # make sure that the matrix data are real; by default, the ham1d
    # package tools construct complex matrices, but we do not need
    # them for diagonalization jobs
    csr = (model.mat.indptr, model.mat.indices, np.real(model.mat.data))

    # create a sparse matrix (AIJ) using a suitable PETSc routine.
    # depending on the choice of communicator comm, the construction
    # can be either parallel or serial.
    matrix = PETSc.Mat().createAIJ(size=(model.nstates, model.nstates),
                                   comm=comm,
                                   csr=csr,
                                   nnz=nnz)

    matrix.setUp()
    # rstart, rend = matrix.getOwnershipRange()
    matrix.assemblyBegin(matrix.AssemblyType.FINAL_ASSEMBLY)
    matrix.assemblyEnd(matrix.AssemblyType.FINAL_ASSEMBLY)

    matrix_sq = matrix.matMult(matrix)
    # ------------------------------------------------------------
    #
    #
    #      GETTING THE HAMILTONIAN TRACE AND TRACE
    #      OF THE HAMILTONIAN SQUARED
    #
    # ------------------------------------------------------------
    # get the square of the matrix from which trace of the
    # squared hamiltonian can be extracted.
    # create a global scatter context which has
    # access to the whole vector contents on all
    # the processes.

    # prepare the vector structure for the
    # diagonal of the hamiltonian array
    # and the diagonal of the square of the
    # hamiltonian array
    diag_H, tmp = matrix.getVecs()
    diag_H_sq, tmp = matrix_sq.getVecs()

    # obtain the diagonals
    diagonals = []
    matrix.getDiagonal(diag_H)
    matrix_sq.getDiagonal(diag_H_sq)

    # prepare the scatter contexts for
    # obtaining the data
    for i, vec in enumerate([diag_H, diag_H_sq]):

        ctx = PETSc.Scatter()
        toall, zvec = ctx.toAll(vec)
        toall.scatterBegin(vec, zvec, 1, 0)
        toall.scatterEnd(vec, zvec, 1, 0)

        # ave ener is needed for selecting
        # the shift and invert target energy
        if i == 0:
            ave_ener = np.mean(zvec)
            print('ave_ener')
            print(ave_ener)

        if mpirank == 0:
            diagonals.append(np.array(zvec))

        ctx.destroy()
        toall.destroy()
        zvec.destroy()

    # ------------------------------------------------------------
    #
    #       DETERMINING THE SPECTRAL EXTREMA
    #
    # ------------------------------------------------------------

    # find the spectral edges needed for the target determination
    E = SLEPc.EPS().create(PETSc.COMM_WORLD)
    E.setOperators(matrix)
    # Hermitian Eigenvalue Problem (HEP)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    # look for the smallest real part eigenvalue
    E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

    E.solve()
    nconv = E.getConverged()
    ener0 = np.real(E.getEigenvalue(0,))

    # look for the largest eigenvalue
    E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    E.solve()

    nconv = E.getConverged()
    ener1 = np.real(E.getEigenvalue(0,))

    if (mpirank == 0):
        print('Emin: {}. Emax: {}'.format(ener0, ener1))

    E.destroy()

    # ------------------------------------------------------------
    #
    #       SHIFT AND INVERT PART
    #
    # ------------------------------------------------------------

    # shift-and-invert part
    E_si = SLEPc.EPS().create(PETSc.COMM_WORLD)
    E_si.setFromOptions()
    E_si.setOperators(matrix)
    E_si.setProblemType(SLEPc.EPS.ProblemType.HEP)

    E_si.setWhichEigenpairs(E_si.Which.TARGET_REAL)
    E_si.setTarget(ave_ener)
    # E_si.setTarget(ener0 + (ener1 - ener0) * 0.5)
    E_si.solve()

    # ------------------------------------------------------------
    #
    #       DESTROY WHEN DONE
    #
    # ------------------------------------------------------------
    sinvert_collect_and_save(model, matrix, E_si, ener0, ener1,
                             mpirank, PETSc, savepath, syspar,
                             modpar, argsDict, syspar_keys,
                             modpar_keys,
                             many_body,
                             save_space)

    E_si.destroy()
    matrix.destroy()
    matrix_sq.destroy()
