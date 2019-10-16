"""
A module with functions for preparing submission scripts
for running jobs either on a SLURM-based cluster or on
the head node/ home machine.

"""


import subprocess as sp

# attributes of the dict:
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
#
programs = {

    'diag': {'name': 'main_diag.py', 'array': True,
             'save': 'Eigvals', 'vectors': True},
    'sff': {'name': 'main_sff.py', 'array': False,
            'save': 'Spectral_stats', 'vectors': False},
    'gaps': {'name': 'main_r.py', 'array': False,
             'save': 'Spectral_stats', 'vectors': False},
    'hdf5': {'name': './utils/hdf5saver.py', 'array': False,
             'save': 'Eigvals', 'vectors': False}

}


def prep_sub_script(mode='diag', queue=False, cmd_arg='',
                    storage='', syspar='',
                    modpar='',
                    slurmargs=["00:00:01",
                               1, 1, 1, 4, 'test', 'log',
                               1, 1],
                    cmd_opt=[],
                    slurm_opt=[],
                    module='',
                    environment='', cd='',
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
            Whether the program is to be executed on a SLURM-based clusterr
            or on the headnode/home machine.
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
        slurm_opt: string
            A string specifying optional arguments to be added to the
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



    Returns:

        submission_scripts: list
            A list of formatted job submission strings/scripts. If
            queue == False and jobs require different seeds, the
            length of the list is greater than one, since a submission
            script is prepared for each seed number. In all other
            cases, only one submission script is prepared.
    """
    prog = programs[mode]
    minseed = slurmargs[7]
    maxseed = slurmargs[8]
    seedlist = ['']

    if prog['array']:
        if queue:
            slurm_opt.append(f'#SBATCH --array={minseed}-{maxseed}')
            seedlist = ['--seed=${SLURM_ARRAY_TASK_ID}']
        else:
            seedlist = [f'--seed={seed}' for seed in
                        range(minseed, maxseed, 1)]

    # The actual command line arguments which are used for running a
    # given program
    tail, head = syspar.split('_pbc_')
    head = 'pbc_' + head
    cmd_scripts = [(
        f"python {prog['name']} {cmd_arg}{' '.join(cmd_opt)} {seed} "
        f"--results={storage}/{prog['save']}/{head}/{tail}/{modpar} "
        f"--syspar={syspar} --modpar={modpar} \n") for seed in
        seedlist]

    modload_script = (
        f"\nmodule purge\n"
        f"module load {module}\n"
        f"conda deactivate\n"
        # f"eval \"$(conda shell.bash hook)\"\n"
        f"conda activate {environment}\n"
    )

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

    misc_script = (
        "\ncd {0}\n"
        "hostname\n"
        "pwd\n"
        "echo $SLURM_CPUS_PER_TASK\n"
        "OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}\n"
    ).format(cd)

    if queue:

        submission_scripts = [sbatch_script + modload_script +
                              misc_script + cmd_script for cmd_script
                              in cmd_scripts]
    else:
        submission_scripts = cmd_scripts

    return submission_scripts


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
                  """#SBATCH --time=00:00:01\n
                  #SBATCH --output=/log/dep_%A.out""")

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
