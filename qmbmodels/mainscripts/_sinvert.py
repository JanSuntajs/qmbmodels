"""
To avoid code repetition between main_sinvert.py
and main_sinvert_short.py routines, we implement the
body of the shift-and-invert code in one wrapper function.


"""
import numpy as np
from scipy import linalg as LA

from ham1d.entropy.rdm import build_rdm
from ham1d.entropy.ententro import Entangled
from qmbmodels.utils.filesaver import savefile


def sinvert_body(mod, argsDict, syspar, syspar_keys,
                 modpar, modpar_keys, mpirank, mpisize, comm,
                 save_metadata, savepath,
                 PETSc, SLEPc):
    """
    A function that prepares the selected quantum hamiltonian
    for a given disorder realization and then performs shift-
    and-invert diagonalization on it in order to extract
    a selected portion of the interior eigenvalues. We currently
    extract the eigenvalues around the hamiltonian's mean energy.

    Parameters:
    -----------
    mod: python module containing the defs.
         of the used hamiltonian (for instance,
         imbrie.py or heisenberg.py)
    argsDict: dict
         Dictionary containing formatted pairs
         of parameter keys and their corresponding
         values which are parsed from the command-line
         arguments.
    syspar: string
         A formatted string containing information about all
         the relevant system parameters.
    syspar_keys: list
         A list of strings indicating which of the keys
         in the argsDict correspond to the system parameters.
    modpar: string
         A formatted string containing information about all
         the relevant model parameters.
    modpar_keys: list
         A list of strings indicating which of the keys in the
         argsDict correspond to the model parameters.
    mpirank: int,
         the rank of the mpi process
    mpisize: int
         the size of the mpi process pool
    comm: mpi communicator used
    save_metadata: boolean,
         Whether the metadata .json dicts are also created
         during the saving part of the code
    savepath: string
         Path to the root of the storage forlder where one
         wants to store the results.

    PETSc, SLEPc: appropriate modules imported from petsc4py and
                  slepc4py, respectively.

    Returns:
    --------
    This function has no return, its output is a .npz
    file containing various quantities of interest which the
    function calculates.

    If save_metadata=True, then also 'metadata' describing the results
    are stored in a created (if yet nonexistent) folder ./metadata
    within the results folder. Metadata are stored in three .json files
    with filenames of the form 'metadata_*.json', 'modpar_*.json' and
    'syspar_*.json'. The form of the results file is quite self-explanatory
    and is as follows:
    savedict = {'Eigenvalues_partial': eigvals,
            'Hamiltonian_diagonal_matelts_partial': diagonals[0],
            'Hamiltonian_squared_diagonal_matelts_partial':
            diagonals[1],
            'Entropy_partial': entropy,
            'Eigenvalues_partial_spectral_info': metadata,
            **fields}
    Hence, we save (in that particular order):
    -> eigenvalues of the partial diagonalization
    -> hamiltonian's diagonal matrix elements
    -> diagonal matrix elements of the squared hamiltonian
    -> entropy of the obtained eigenstates (for many-body calculations)
    -> ipr (inverse participation ratio) (for free or anderson calculations)
    -> metadata (details about the calculations, procedures used,
    hamiltonian models etc.)
    -> fields -> a dictionary containing information about the
    -> random fields used for different realizations of disorder.
    """
    # ------------------------------------------------------------
    #
    #       ARRAY CONSTRUCTION
    #
    # ------------------------------------------------------------

    # distinguish two special cases
    cond_anderson = (argsDict['ham_type'] == 'anderson')
    cond_free = (argsDict['ham_type'] == 'free1d')

    many_body = not (cond_free or cond_anderson)
    # get the instance of the appropriate hamiltonian
    # class and the diagonal random fields used
    model, fields = mod.construct_hamiltonian(argsDict, parallel=True,
                                              mpirank=mpirank,
                                              mpisize=mpisize)
    print(fields.keys())
    for key in fields.keys():
        oldkey = key
        if '_partial' not in key:
            fields[key + '_partial'] = fields.pop(oldkey)
    print('fields:')
    print(fields)
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

    nconv = E_si.getConverged()
    print('nconv is:')
    print(nconv)
    # ------------------------------------------------------------
    #
    #       EXTRACTION OF EIGENVECTORS AND EIGENVALUES
    #       ENTANGLEMENT ENTROPY CALCULATION
    #       IPR CALCULATION FOR NON-MANY BODY MODELS
    #       IPR - INVERSE PARTICIPATION RATIO
    #
    # ------------------------------------------------------------

    # store to a numpy array
    if nconv > 0:

        # for finding the eigenvectors

        vr, tmp = matrix.getVecs()
        vi, tmp = matrix.getVecs()

        # if mpirank == 0:

        eigvals = []
        if many_body:
            entropy = []
        else:
            ipr = []

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
                eigvals.append(eigval)
                eigvec = np.array(zvec)

                if many_body:
                    if model.Nu is not None:
                        rdm_matrix = build_rdm(eigvec, int(
                            argsDict['L'] / 2.), argsDict['L'], int(
                            argsDict['nu']))
                        rdm_eigvals = LA.eigvalsh(rdm_matrix.todense())
                        rdm_eigvals = rdm_eigvals[rdm_eigvals > 1e-014]
                        entro = -np.dot(rdm_eigvals, np.log(rdm_eigvals))
                    else:
                        entangled = Entangled(eigvec, argsDict['L'],
                                              int(argsDict['L'] / 2.))
                        entangled.partitioning('homogenous')
                        entangled.svd()
                        entro = entangled.eentro()

                    entropy.append(entro)
                else:

                    ipr_state = np.sum(np.abs(eigvec)**4)
                    ipr.append(ipr_state)

            # destroy the scatter context before the new
            # loop iteration
            ctx.destroy()
            tozero.destroy()
            zvec.destroy()

        if mpirank == 0:
            # -----------------------------------------

            sortargs = np.argsort(eigvals)
            eigvals = np.array(eigvals)[sortargs]

            metadata = np.array([ener0, ener1, nconv])

            saveargs = (savepath, syspar, modpar, argsDict,
                        syspar_keys, modpar_keys, 'partial')

            savedict = {'Eigenvalues_partial': eigvals,
                        'Hamiltonian_diagonal_matelts_partial': diagonals[0],
                        'Hamiltonian_squared_diagonal_matelts_partial':
                        diagonals[1],
                        'Eigenvalues_partial_spectral_info': metadata,
                        ** fields}

            if many_body:
                entropy = np.array(entropy)[sortargs]
                savedict['Entropy_partial'] = entropy
            else:
                ipr = np.arry(entropy)[sortargs]
                savedict['Ipr_partial'] = ipr
            # save the eigenvalues, entropy and spectral info as
            # a npz array

            savefile(savedict, *saveargs, save_metadata, save_type='npz')
            save_metadata = False

    E_si.destroy()
    matrix.destroy()
    matrix_sq.destroy()
