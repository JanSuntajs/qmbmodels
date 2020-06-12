from qmbmodels.senderscripts.batchsend import BatchSender
from qmbmodels.utils.cmd_parser_tools import mode_parser

from qmbmodels.models.prepare_model import import_model
import numpy as np

if __name__ == '__main__':
    storage = '.'
    ham_type = 'spin1d'
    model = 'imbrie'
    mod = import_model(model)

    params = {
        'L': [8],
        'J': [0.8],
        'dJ': [0.2],
        'H': [0.],
        'dH': [20.],  # np.arange(1., 9., 1.),  # np.arange(1., 9, 0.5),
        'Gamma': [1.],
        'dGamma': [0.],
        'pbc': [False],
        'ham_type': [ham_type],
        'disorder': ['single'],
        'min_seed': [48],
        'max_seed': [58],
        'step_seed': [5],
        'sff_min_tau': [-5],
        'sff_max_tau': [2],
        'sff_n_tau': [30],
        'sff_eta': [0.5],
        'sff_unfolding_n': [3],
        'sff_filter': ['gaussian'],
        'r_step': [0.05],
    }

    # shift and invert parameters
    sinvert_params = ['--model={}'.format(model),
                      '-eps_type krylovschur', '-eps_nev 32',
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

    name = f'{model}_{ham_type}_test'
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
