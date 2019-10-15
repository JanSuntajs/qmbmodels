from senderscripts.batchsend import BatchSender
from utils.cmd_parser_tools import mode_parser

from models import heisenberg as mod

if __name__ == '__main__':

    model = 'spin1d'
    params = {
        'L': [12],
        'nu': [6],
        'J1': [1.],
        'J2': [1.],
        'W': [0.],
        'dW': [3.],
        'delta1': [.55],
        'delta2': [.55],
        'pbc': [True],
        'ham_type': ['ferm1d'],
        'disorder': ['uniform', 'gaussian', 'binary'],
        'min_seed': [1],
        'max_seed': [5],
        'sff_min_tau': [-5],
        'sff_max_tau': [2],
        'sff_n_tau': [300],
        'sff_eta': [0.5],
        'sff_unfolding_n': [3],
        'sff_filter': ['gaussian'],
        'r_step': [0.05],
    }

    syspar_keys = mod.syspar_keys
    modpar_keys = mod.modpar_keys
    auxpar_keys = ['sff_min_tau', 'sff_max_tau', 'sff_n_tau', 'sff_eta',
                   'sff_unfolding_n', 'sff_filter', 'r_step']

    time = "00:59:59"
    nodes = 1   # number of nodes
    ntasks = 1  # number of threads
    memcpu = 4  # memory in GB per CPU!

    queue, mode = mode_parser()

    print(f"{mode}")
    # args = arg_parser(syspar_keys, modpar_keys)

    sender = BatchSender(params, syspar_keys, modpar_keys, auxpar_keys)

    sender.run_jobs(mode, queue=queue, time=time, nodes=nodes, ntasks=ntasks,
                    memcpu=memcpu)
