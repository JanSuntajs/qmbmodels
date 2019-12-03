"""
Module which takes care
of the initialization
of the parallel libraries
petsc and slepc.


This code is intended to be used
by using the following import
line in the main program:

from utils.set_petsc_slepc import *
"""

import sys
import slepc4py

# initalize the slepc
# options from the command-line
# arguments
slepc4py.init(sys.argv)

# import PETSc and SLEPc
# libraries
from petsc4py import PETSc
from slepc4py import SLEPc
