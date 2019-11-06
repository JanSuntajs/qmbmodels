"""
A module with functions for preparing submission scripts
for running jobs either on a SLURM-based cluster or on
the head node/ home machine.

"""


import subprocess as sp

# attributes of the programs dict:
# programs[mode] -> which type of job to perform
# programs[mode]['name'] -> name of the executable script
# programs[mode]['array'] -> if the job is meant to be
#                            performed as an array
# programs[mode]['save'] -> which name to use when saving
#                           the output results
# programs[mode]['vectors'] -> whether the job can also
#                              give eigenvectors (large
#                              and potentially memory-
#                              consuming arrays) as output.
# programs[mode]['noqueue'] -> if job is never meant to be
#                              ran on the cluster, such as
#                              in the case of 'folder' program
#                              which only creates the folder
#                              structure.
# programs[mode]['mpi'] -> whether the job can be executed in
#                          parallel using the mpi paradigm
#
# What different modes do:
#
#   diag -> full diagonalization
#   sff -> spectral form factor calculation
#   gaps -> <r> value calculation
#   hdf5 -> a helper job that is run after the diag
#           or sinvert jobs have finished ->
#           collects the eigenvalue calculation results
#           into a single hdf5 file with multiple
#           datasets. The user does not need to
#           call the routine, it is called automatically
#           after the diagonalization jobs to do the
#           'cleanup'.
#   folder -> a helper routine that takes care of the
#             appropriate directory structure creation.
#             This mode is never run on the cluster since
#             queueing in order to perform such a trivial
#             task would be wasteful.
#
programs = {

    'diag': {'name': './main_diag.py', 'array': True,
             'save': 'Eigvals', 'vectors': True,
             'noqueue': False, 'mpi': False},
    'sff': {'name': './main_sff.py', 'array': False,
            'save': 'Spectral_stats', 'vectors': False,
            'noqueue': False, 'mpi': False},
    'gaps': {'name': './main_r.py', 'array': False,
             'save': 'Spectral_stats', 'vectors': False,
             'noqueue': False, 'mpi': False},
    'gaps_partial': {'name': './main_r_partial.py', 'array': False,
                     'save': 'Spectral_stats', 'vectors': False,
                     'noqueue': False, 'mpi': False},
    'hdf5': {'name': './utils/hdf5saver.py', 'array': False,
             'save': 'Eigvals', 'vectors': False,
             'noqueue': False, 'mpi': False},
    'folder': {'name': './utils/prepfolders.py', 'array': False,
               'save': None, 'array': False, 'noqueue': True,
               'mpi': False},
    'sinvert': {'name': './main_sinvert.py', 'array': True,
                'save': 'Eigvals', 'vectors': True,
                'noqueue': False, 'mpi': True}

}

# diagonalization modes
diag_modes = ['diag', 'sinvert']
# define shift-and-invert keys:
#
# eps_type -> eps_solver
# eps_nev -> number of eps eigenvalues
# st_type -> type of the spectral transform used
# st_ksp_type
# st_pc_type -> preconditioner type
# st_pc_factor_mat_solver_type -> which solver to use
#

sinvert_keys = ['eps_type', 'eps_nev', 'st_type', 'st_ksp_type',
                'st_pc_type',
                'st_pc_factor_mat_solver_type',
                'mat_mumps_icntl']


def prep_sub_script(mode='diag', queue=False,
                    interactive=False, cmd_arg='',
                    storage='', syspar='',
                    modpar='',
                    slurmargs=["00:00:01",
                               1, 1, 1, 4, 'test', 'log',
                               1, 1],
                    cmd_opt=[],
                    slurm_opt=[],
                    module='',
                    environment='', cd='',
                    name=''
                    ):
    """

    A function that prepares a script that can be run either
    on a SLURM-based cluster or on the headnode/home machine.

    An example SLURM-ready script would be:

        ##  SLURM-BLOCK: this part contains SLURM
        ##  commands for allocation of time, cpus, memory
        ##  etc. -> sbatch_script

            #!/bin/bash
            #SBATCH --time=00:00:01  # format: DD-HH:MM:SS
            #SBATCH --nodes=1  # number of nodes
            #SBATCH --ntasks=1  # number of tasks
            #SBATCH --cpus-per-task=1  # number of cpus per a given task
            #SBATCH --mem-per-cpu=4  #  memory (in GB) per cpu
            #SBATCH --job-name 'pretty_name'  #  give job a human-readable name
            #SBATCH --output=log_folder/slurm_%A_%a.out  #  output files

        ##  MODLOAD_SCRIPT: in order for the scripts to run
        ##  properly/as expected,
        ##  one should first purge the preloaded modules and then load the
        ##  ones for which the code was prepared/tested. The procedure is
        ##  as follows:

            module purge
            module load <module name>  ## for instance: Anaconda3/5.3.0
            conda activate <environment name>

        ##  MISC_SCRIPT: change current directory if needed, display hostname
        ##  and current working directory, display the number of cpus per
        ##  task allocated in SLURM and set the OMP_NUM_THREADS environment
        ##  variable, which is important for multiprocessing jobs.

            cd .
            hostname
            pwd
            echo $SLURM_CPUS_PER_TASK

            OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

        ##  CMD_SCRIPT: run the executable program and parse the command-line
        ##  arguments
            ./imbrie_main_sff.py {8} --results={9}/Spectral_stats/{10}/{11} \
            --syspar={10} --modpar={11}

    NOTE:
    if queue == False,
    then only the CMD_SCRIPT part is generated.

    Parameters:

        mode: string
            Specifier for which program type to run. Currently 'diag' and
            'sff' are supported and the default is set to 'diag'.
        queue: boolean
            Whether the program is to be executed on a SLURM-based cluster
            or on the headnode/home machine.
        interactive: boolean
            Whether the program is to be executed on a SLURM-based cluster's
            interactive mode.
        cmd_arg: string
            A string which follows the name of the executable program when
            run in the command line:
            ./<executable_name> <cmd_arg>
            NOTE: the contents of this string should be related to the
            structure of the considered Hamiltonian and should thus contain
            information such as system size, dimensionality, values of the
            exchange couplings etc.
        cmd_opt: list
            A list of strings specifying optional command-line arguments.
            This is implemented for flexibility - most of the commandline
            arguments match for the majority of our programs, so a single
            <cmd_arg>
            string often suffices, but there can be differences. This option
            allows to add other command-line possibilities without the need
            to restructure this code in the future.
            Note that this function already
            takes care of the cases when the seed needs to be added manually
            when code is not executed on a SLURM-based cluster or when the job
            does not need a seed altogether, such as in the case of sff jobs.
            Syntax:
            ./<executable_name> <cmd_arg> <cmd_opt>
        slurm_opt: list
            A list specifying optional arguments to be added to the
            SLURM-related part of the script. Much as in the cmd_opt case,
            this part is included for flexibility. All other parameters
            remaining equal, we simply append the required additional
            set of #SBATCH commands to the preexisting core set. As with
            the cmd_opt parameter, the function already addresses the cases
            related to the usage of array jobs.
        storage: string
            Path to where the results should be stored.
        modpar, syspar: strings
            Formatted strings specifying model and system parameters.
        module: string
            Which software module to load (on the spinon cluster
            which supports modules)
        environment: string
            Which (ana)conda environment to load.
        cd: string
            Whether to change the working environment or not. The default
            option leaves the working directory as it is.
        slurmargs: list
            A list of arguments to be provided to the SLURM-BLOCK part
            of the script. Order:
            [<time>, <nodes>, <ntasks>, <cpus-per-task>,<mem-per-cpu<,
            <job-name>, <output>, <minseed>, <maxseed>]
        name: str
            A human readable description of the job which also
            serves as the top folder in the results directory.



    Returns:

        submission_scripts: list
            A list of formatted job submission strings/scripts. If
            queue == False and jobs require different seeds, the
            length of the list is greater than one, since a submission
            script is prepared for each seed number. In all other
            cases, only one submission script is prepared.
    """
    slurm_opt = slurm_opt.copy()
    prog = programs[mode]
    minseed = slurmargs[7]
    maxseed = slurmargs[8]
    seedlist = ['']

    if prog['array']:
        if queue:
            slurm_opt.append(f'#SBATCH --array={minseed}-{maxseed}')
            seedlist = ['--seed=${SLURM_ARRAY_TASK_ID}']
        elif (interactive or (not queue)):
            seedlist = [f'--seed={seed}' for seed in
                        range(minseed, maxseed, 1)]

    # The actual command line arguments which are used for running a
    # given program
    tail, head = syspar.split('_pbc_')
    head = 'pbc_' + head
    results = f"{storage}/{name}/{head}/{tail}/{modpar}"

    if prog['mpi']:
        if queue or interactive:
            nproc = '${SLURM_NTASKS}'
        else:
            nproc = slurmargs[2]
        execname = 'mpiexec -np {} python'.format(nproc)
        # execname = 'mpiexec python'
        cmd_opt_ = cmd_opt.copy()
    else:
        execname = 'python'
        # exclude command options referring to the
        # mpi jobs in this case
        cmd_opt_ = [opt for opt in cmd_opt if not any(
            [key in opt for key in sinvert_keys])]

    cmd_scripts = [(
        f"{execname} {prog['name']} {cmd_arg} {' '.join(cmd_opt_)} {seed} "
        f"--results={results} "
        f"--syspar={syspar} --modpar={modpar} \n") for seed in
        seedlist]

    modload_script = (
        f"\nmodule purge\n"
        f"module load {module}\n"
        f"conda deactivate\n"
        # f"eval \"$(conda shell.bash hook)\"\n"
        f"conda activate {environment}\n"
    )

    # job allocation parameters for running on the SLURM cluster
    sbatch_script = (
        "#!/bin/bash"
        "\n#SBATCH --time={0}\n"
        "#SBATCH --nodes={1}\n"
        "#SBATCH --ntasks={2}\n"
        "#SBATCH --cpus-per-task={3}\n"
        "#SBATCH --mem-per-cpu={4}\n"
        "#SBATCH --job-name={5}\n"
        "#SBATCH --output={6}slurm_%A_%a.out\n"
        "{7}\n").format(*slurmargs[:7], '\n'.join(slurm_opt))

    # allocation parameters for running on
    interactive_script = (
        "srun --time={0} --nodes={1} --ntasks={2} "
        "--cpus-per-task={3} --mem-per-cpu={4} "
        "--job-name={5} --output={6}slurm_%A_%a.out "
        "{7} --pty bash -i").format(*slurmargs[:7], ' '.join(slurm_opt))

    misc_script = (
        "\ncd {0}\n"
        "hostname\n"
        "pwd\n"
        "echo $SLURM_CPUS_PER_TASK\n"
        "OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}\n"
    ).format(cd)

    runscript = ""
    if (queue and not interactive):

        submission_scripts = [sbatch_script + modload_script +
                              misc_script + cmd_script for cmd_script
                              in cmd_scripts]
    elif interactive:

        submission_scripts = modload_script + misc_script
        for cmd_script in cmd_scripts:
            submission_script += cmd_script

        runscript = interactive_script
    else:
        submission_scripts = cmd_scripts

    return submission_scripts, runscript


def prepare_dependency_script(scripts, name, *args, **kwargs):
    """
    In order to be able to run our jobs consecutively,
    we make use of the SLURM dependencies. See this
    URL to find out more about using them:

    https://hpc.nih.gov/docs/job_dependencies.html
    (accessed on: 29/07/2019)

    NOTE: currently only afterany option is implemented.
    If needed, we could add an additional list of dependencies
    for consecutive jobs.

    """

    dep_script = ("""#!/bin/bash\n"""
                  """#SBATCH --time=00:00:01\n"""
                  """#SBATCH --output=./log/dep_%j.out\n""")

    dep_script += f"\njid0=$(sbatch --parsable {scripts[0]})"
    try:
        for i, script in enumerate(scripts[1:]):
            j = i + 1
            dep_script += (f"\njid{j}=$(sbatch --parsable "
                           f"--dependency=afterany:"
                           f"$jid{i} {script})")
    except IndexError:
        pass

    with open(name, 'w') as namescript:

        namescript.write(dep_script)

    return name
