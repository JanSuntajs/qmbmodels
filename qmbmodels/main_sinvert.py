#!/usr/bin/env python

"""
This code implements the PETSc and SLEPc
libraries in order to partially diagonalize
the quantum model hamiltonian of choice. While
the general structure of the code allows for choice
of different spectral transformations and different
solver contexts, the main intent of this code is to
obtain the eigenvalues and the eigenvectors from
a selected region of the hamiltonian's spectrum
and to also calculate the eigenvector's entanglement
entropy.


"""

import sys
import slepc4py
from utils import set_mkl_lib

slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
from scipy import linalg as LA

from ham1d.entropy.rdm import build_rdm
from ham1d.entropy.ententro import Entangled
from utils.cmd_parser_tools import arg_parser
from utils.filesaver import savefile
from models.prepare_model import select_model

store_eigvecs = False

if __name__ == '__main__':

    mod, model_name = select_model()

    comm = PETSc.COMM_WORLD

    mpisize = comm.size
    mpirank = comm.rank

    syspar_keys = mod.syspar_keys
    modpar_keys = mod._modpar_keys

    argsDict, extra = arg_parser(syspar_keys, modpar_keys)
    argsDict['model'] = model_name
    syspar_keys.append('model')

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    print('Using seed: {}'.format(argsDict['seed']))

    # ------------------------------------------------------------
    #
    #       ARRAY CONSTRUCTION
    #
    # ------------------------------------------------------------

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

    if(mpirank == 0):

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

    # ------------------------------------------------------------
    #
    #       EXTRACTION OF EIGENVECTORS AND EIGENVALUES
    #       ENTANGLEMENT ENTROPY CALCULATION
    #
    # ------------------------------------------------------------

    # store to a numpy array
    if nconv > 0:

        # for finding the eigenvectors

        vr, tmp = matrix.getVecs()
        vi, tmp = matrix.getVecs()

        # if mpirank == 0:

        eigvals = []
        entropy = []

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

                if model.Nu is not None:

                    rdm_matrix = build_rdm(eigvec, int(
                        argsDict['L'] / 2.), argsDict['L'], int(
                        argsDict['L'] / 2.))
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

            # destroy the scatter context before the new
            # loop iteration
            ctx.destroy()
            tozero.destroy()
            zvec.destroy()
            # -----------------------------------------

        if mpirank == 0:

            sortargs = np.argsort(eigvals)
            eigvals = np.array(eigvals)[sortargs]

            entropy = np.array(entropy)[sortargs]
            metadata = np.array([ener0, ener1, nconv])

            saveargs = (savepath, syspar, modpar, argsDict,
                        syspar_keys, modpar_keys, 'partial')

            savedict = {'Eigenvalues_partial': eigvals,
                        'Hamiltonian_diagonal_matelts_partial': diagonals[0],
                        'Hamiltonian_squared_diagonal_matelts_partial':
                        diagonals[1],
                        'Entropy_partial': entropy,
                        'Eigenvalues_partial_spectral_info': metadata,
                        **fields}
            # save the eigenvalues, entropy and spectral info as
            # a npz array
            savefile(savedict, *saveargs, True, save_type='npz')

    E_si.destroy()
    matrix.destroy()
    matrix_sq.destroy()
