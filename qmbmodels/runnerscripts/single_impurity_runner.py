import numpy as np
from scipy.special import comb

from qmbmodels.senderscripts.batchsend import BatchSender
from qmbmodels.utils.cmd_parser_tools import mode_parser

from qmbmodels.models.prepare_model import import_model


if __name__ == '__main__':
    # storage = './../../../Entanglement_entropy_numerical_results'
    storage = './..'
    ham_type = 'spin1d'
    model = 'heisenberg_single_impurity'
    mod = import_model(model)
    params = {
        'L': [12],
        'J': [4.],
        'dJ': [0.0],
        'W': [0.],
        'dW': [0.0],
        # [0.00001, 0.00005, 0.00010, 0.00050, 0.00100, 0.00500,
        'noise': [0.00000],
        # 0.01000, 0.05000, 0.10000, 0.50000, 1.00000],
        # [np.cos(np.pi / x) for x in np.arange(1, 7, 1)],
        'delta': [np.cos(np.pi / x) for x in np.arange(1, 7, 1)], #np.arange(0, 1.01, 0.01),
        'ddelta': [0.0],
        'pbc': [False],
        'ham_type': [ham_type],
        'disorder': ['single'],
        'min_seed': [1],
        'max_seed': [1],
        'step_seed': [1],
        'sff_min_tau': [-5],
        'sff_max_tau': [2],
        'sff_n_tau': [30],
        'sff_eta': [0.5],
        'sff_unfolding_n': [3],
        'sff_filter': ['gaussian'],
        'r_step': [0.05],
    }

    params['nu'] = [int(val * 0.5) for val in params['L']]
    params['sff_min_tau'] = [np.log10(1 / comb(val, 0.5 * val))
                             for val in params['L']]
    # shift and invert parameters
    sinvert_params = ['--model={}'.format(model),
                      '-eps_type krylovschur', '-eps_nev 500',
                      '-st_type sinvert',
                      '-st_ksp_type preonly',
                      '-st_pc_type lu',
                      '-st_pc_factor_mat_solver_type mumps',
                      '-mat_mumps_icntl_28 2',
                      '-mat_mumps_icntl_29 2']

    syspar_keys = mod.syspar_keys
    modpar_keys = mod.modpar_keys
    auxpar_keys = ['sff_min_tau', 'sff_max_tau', 'sff_n_tau', 'sff_eta',
                   'sff_unfolding_n', 'sff_filter', 'r_step']

    name = f'{model}_{ham_type}_check_degs_special_points'
    time = "00:59:59"
    nodes = 1   # number of nodes
    ntasks = 1  # number of threads
    memcpu = 4  # memory in GB per CPU!
    cputask = 1

    slurm_opt = []
    queue, mode = mode_parser()

    print(f"{mode}")
    # args = arg_parser(syspar_keys, modpar_keys)

    sender = BatchSender(params, syspar_keys, modpar_keys,
                         auxpar_keys, cmd_opt=sinvert_params,
                         storage=storage, slurm_opt=slurm_opt)

    sender.run_jobs(mode, queue=queue,
                    time=time, nodes=nodes, ntasks=ntasks,
                    memcpu=memcpu, name=name, sourcename='petscenv',
                    cputask=cputask)
