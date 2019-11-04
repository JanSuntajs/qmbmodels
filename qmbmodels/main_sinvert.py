#!/usr/bin/env python

import sys
import slepc4py

slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
from models import heisenberg
from utils.cmd_parser_tools import arg_parser
from utils.filesaver import savefile

if __name__ == '__main__':

    comm = PETSc.COMM_WORLD

    mpisize = comm.size
    mpirank = comm.rank

    syspar_keys = heisenberg.syspar_keys
    modpar_keys = heisenberg._modpar_keys

    argsDict, extra = arg_parser(syspar_keys, modpar_keys)

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    print('Using seed: {}'.format(argsDict['seed']))

    # get the instance of the appropriate hamiltonian
    # class and the diagonal random fields used
    model, fields = heisenberg.construct_hamiltonian(argsDict, parallel=True,
                                                     mpirank=mpirank,
                                                     mpisize=mpisize)

    # prepare for parallel PETSc assembly
    nnz = (model._d_nnz, model._o_nnz)

    # make sure that the matrix data are real; by default, the ham1d
    # package tools construct complex matrices, but we do not need
    # them for diagonalization jobs
    csr = (model.mat.indptr, model.mat.indices, np.real(model.mat.data))

    matrix = PETSc.Mat().createAIJ(size=(model.nstates, model.nstates),
                                   comm=comm,
                                   csr=csr,
                                   nnz=nnz)

    matrix.setUp()
    rstart, rend = matrix.getOwnershipRange()
    matrix.assemblyBegin(matrix.AssemblyType.FINAL_ASSEMBLY)
    matrix.assemblyEnd(matrix.AssemblyType.FINAL_ASSEMBLY)

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

    # shift-and-invert part
    E_si = SLEPc.EPS().create(PETSc.COMM_WORLD)
    E_si.setFromOptions()
    E_si.setOperators(matrix)
    E_si.setProblemType(SLEPc.EPS.ProblemType.HEP)

    E_si.setWhichEigenpairs(E_si.Which.TARGET_REAL)
    E_si.setTarget(ener0 + (ener1 - ener0) * 0.5)
    E_si.solve()

    nconv = E_si.getConverged()

    # store to a numpy array
    if mpirank == 0:

        eigvals = []

        for i in range(0, nconv):

            eigval = E_si.getEigenvalue(i)

            eigvals.append(np.real(eigval))

        eigvals = np.array(np.sort(eigvals))
        print('eigvals:')
        print(eigvals)

        savefile(eigvals, savepath, syspar, modpar, argsDict,
                 syspar_keys, modpar_keys, 'partial', True)

    E_si.destroy()
    matrix.destroy()
