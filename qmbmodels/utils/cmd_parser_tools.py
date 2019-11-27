
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mode_parser():
    """
    Parse the --queue optional argument
    given to the main script.

    To run the code on the cluster, use
    the main script in the following way:

    ./slurmrunner.py --queue

    Leave the --queue optional argument if
    the code is not meant to be run on the
    cluster.
    """

    parser = argparse.ArgumentParser(prog='Run code on a slurm-based cluster '
                                     'or on the head node/home machine and '
                                     'specify which task to execute.')

    parser.add_argument("-q", "--queue",
                        help="Whether the job is ran in the slurm "
                        "queue or not.", action="store_true")

    parser.add_argument("--eigvecs",
                        help="Whether the diagonalization job should "
                        "also calculate the eigenvectors.",
                        action="store_true")

    parser.add_argument("mode", metavar='mode',
                        help="Which kind of task to perform",
                        nargs='*', default=['diag'],
                        )

    args, extra = parser.parse_known_args()

    print(args)
    print(extra)
    return args.queue, args.mode


def arg_parser(system_keys, module_keys):
    """
    Get the values of the commmand line
    arguments which are to be given to the
    main script.
    """

    parser = argparse.ArgumentParser(prog='Obtain runtime command arguments '
                                     'for the executable script.')

    spec_system_keys = ['pbc', 'disorder', 'ham_type', 'model']
    for key in system_keys:
        if key not in spec_system_keys:
            type_ = int
            default = 0
            parser.add_argument('--{}'.format(key),
                                type=type_, default=default)
        else:
            # pbc or obc (periodic or open boundary conditions)
            if key == 'pbc':
                parser.add_argument('--{}'.format(key),
                                    type=str2bool, default=True)
            # select the disorder type
            if key == 'disorder':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='none')

            # select the hamiltonian type ->
            # can be spin1d, ferm1d, or free
            if key == 'ham_type':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='spin1d')
            # select the actual physical model, such as
            # the heisenberg or imbrie model
            if key == 'model':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='')

    for key in module_keys:

        if 'seed' in key:
            argtype = int
        else:
            argtype = float

        parser.add_argument('--{}'.format(key),
                            type=argtype, default=argtype(0.0))

    for name in ['--results', '--syspar', '--modpar']:
        parser.add_argument(name, type=str, default='.')

    args, extra = parser.parse_known_args()
    return vars(args), extra


def arg_parser_general(*args):
    """
    A general function for parsing the command-line arguments.

    Parameters
    ----------

    args: a tuple of dictionaries, each of the form:

    {'cmd_arg': [type_, default_value],}

    Here, 'cmd_arg' key specifies the command-line argument to be
    parsed and [type_, default_value] list specifies the type
    of the argument to be parsed and default_value specifies
    the default value which should be given if no value is parsed.


    """

    parser = argparse.ArgumentParser(prog='Obtain runtime command arguments '
                                     'for the executable script.')

    for arg in args:

        for key, value in arg.items():

            type_, default = value

            parser.add_argument('--{}'.format(key),
                                type=type_, default=default)

    args, extra = parser.parse_known_args()

    return vars(args), extra
