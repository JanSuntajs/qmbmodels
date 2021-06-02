"""
Contains routines for parsing
the command-line arguments.

"""

import argparse

from qmbmodels.models._common_keys import comm_syspar_keys

def str2bool(v):
    """
    Converts a string to a
    boolean.

    Parameters:
    -----------

    v: {boolean, string}
       A string to be converted to a
       boolean. If v is already
       boolean, no conversion occurs.
       If v is a string, the following
       values are converted to True:
       ('yes', 'true', 't', 'y', '1')
       The following values are converted
       to False:
       ('no', 'false', 'f', 'n', '0')
       Though the lists above are written
       in lower-case, the input is
       case-insensitive.

    Returns:
    --------
    boolean

    Raises:
    -------

    argparse.ArgumentTypeError:
        If v is not boolean or in one of the
        lists specified above.
    """
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
    Parse which programs should be invoked
    once the main script is called.

    The function allows the user to specify whether
    the code should be executed on a SLURM-based cluster
    or on the head node/home machine. It also allows the
    user to specify which tasks to execute (for instance,
    diagonalization followed by calculation of some spectral
    observables.)

    The function makes sure that the following command-line
    arguments are parsed:

    Positional arguments:
    mode: list
          Contains a list of strings which specify which
          tasks to perform. Entries of the list can be:
          'diag', 'sinvert', 'sinvert_short', 'gaps', 'gaps_partial',
          'hdf5', 'folder'.
          If mode is not specified, the default value ['diag']
          is assumed which would invoke a diagonalization
          procedure.

    Optional arguments:
    "-q" or "--queue" (short and long form, respectively):
        Add this argument to your main script if the latter
        is to be performed on a SLURM-based cluster. Ommit
        othervise. Ordering the program to run jobs on a
        cluster is thus achieved as follows:
        python <mainscript.py> --queue ...(some other commands)
        where <mainscript.py> is our main script.

    "--eigvecs": whether the diagonalization job should also
                 calculate the eigenvectors. NOTE: for now,
                 this functionality has not yet been made
                 functional so adding the "--eigvecs" command-line
                 argument really has no effect whatsoever.

    Returns
    -------

    args.queue: boolean
                Whether the job is to be performed on a cluster (if True)
                or not.
    args.mode: list
               A list of strings that corresponds to the mode command-line
               argument.
    """

    # provide a description
    parser = argparse.ArgumentParser(prog='Run code on a slurm-based cluster '
                                     'or on the head node/home machine and '
                                     'specify which task to execute.')

    # if -q or --queue cmd-arg is present, store its value as true
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

    spec_system_keys = comm_syspar_keys + ['model']
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
            # whether to reduce storage space by
            if key == 'save_space':
                parser.add_argument('--{}'.format(key),
                                    type=str2bool, default=False)
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


def arg_parser_general(args):
    """
    A general function for parsing the command-line arguments.

    Parameters
    ----------

    args: a dictionary of the form:

    {'cmd_arg': [type_, default_value],}

    Here, 'cmd_arg' key specifies the command-line argument to be
    parsed and [type_, default_value] list specifies the type
    of the argument to be parsed and default_value specifies
    the default value which should be given if no value is parsed.


    """

    parser = argparse.ArgumentParser(prog='Obtain runtime command arguments '
                                     'for the executable script.')

    for key, value in args.items():

        type_, default = value

        parser.add_argument('--{}'.format(key),
                            type=type_, default=default)

    args, extra = parser.parse_known_args()

    return vars(args), extra
