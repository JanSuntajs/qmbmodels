
from . import set_mkl_lib

# this routines only work with numba dev for now!
try:
    from numba import get_num_threads, set_num_threads

    n_cores = set_mkl_lib.mkl_get_max_threads()

    # set the number of numba cores
    set_num_threads(n_cores)
    print(('This program relies on '
           'numba parallelism. We thus set'
           ' the OMP_NUM_THREADS to 1 and '
           'set the NUMBA_NUM_THREADS instead.'))

    mkl_rt.mkl_set_max_threads(1)
    print(f'NUMBA_NUM_THREADS: {get_num_threads()}')


except ImportError:
    print(('WARNING: get_num_threads() '
           'and set_num_threads() numba '
           'are only available in dev mode!'))
