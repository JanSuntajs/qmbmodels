
try:
    import ctypes  # ctypes, used for determining the mkl num threads

    mkl_present = True
    mkl_string = 'mkl_'
    mkl_rt = ctypes.CDLL('libmkl_rt.so')

    def mkl_set_num_threads(cores):
        # Set # of MKL threads
        mkl_rt.MKL_Set_Num_Threads(cores)

    def mkl_get_max_threads():
        # # of used MKL threads
        print(mkl_rt.MKL_Get_Max_Threads())

    print('mkl maximum number of threads is:')
    mkl_get_max_threads()


except OSError:
    mkl_string = 'no_mkl_'
    mkl_present = False
    print(('WARNING: No libmkl_rt shared object! The program'
           'will not use mkl routines and optimizations!'))
