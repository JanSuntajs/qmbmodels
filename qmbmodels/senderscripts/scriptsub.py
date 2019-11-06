"""
A module with a definition of the SubmittedScript
class which takes care of preparation of the
scripts to be submitted to the SLURM-based
clusters as well as for handling jobs that are
run on the head node or on the home machine.


"""


import os
import subprocess as sp

from . import runscripts as rsc


class SubmittedScript(object):

    """
    A class for creating scripts o be submitted
    for running either on SLURM-based clusters or
    on the head node/home machine.

    Attributes:

    modes: list
            A list of strings specifying in which mode
            the job should be executed. Currently
            supported modes are 'diag' and 'sff' which
            account for diagonalization and sff calculation
            jobs, respectively.
    queue: boolean
            Whether the job is to be ran on the cluster or
            not.
    calc_vectors: boolean
            Whether the (diagonalization) job is supposed
            to calculate the eigenvectors as well.
    time: string
            Time in the format "DD-HH:MM:SS" -> an estimate
            of the job's duration if the code is ran on the
            SLURM-based cluster.
    nodes: int
            Number of nodes on which the job is executed if
            ran on SLURM.
    ntasks, cputask: int
            number of tasks for the job and number
            of cpus per task. Using these two options, one
            can specify the total number of cpus needed for
            the job.
    memcpu: int
            Memory (in GB) required for the job.
    name: string
            A human readable job descriptor.
    module: string,
            in case when code is ran on a SLURM-based cluster.
            one should also specify which software module to load
            in case that the environment variables on the headnode
            and on the actual compute nodes do not match. The
            preferred way of running environment-dependent jobs
            is to include lines
            module purge
            module load <required modules>
            source/conda activate <preferred environment>
            in your sbatch submission script.
    cd:     string
            A string specifying
    scripts: list
            In case self.queue == True, scripts is an attribute
            which stores a list of scripts (e.g. multiline strings
            with instructions for running programs on a SLURM-based
            cluster) to be executed.

    Methods:

    prepare_script(self, job, sender)
            A routine that prepares a list of scripts to be executed
            on a SLURM-based cluster if self.queue == True. The list
            is then sent to the cluster and executed after the
            self._send() command is issued. In the opposite case,
            the jobs are executed in a sequential manner directly
            without calling the self._send() command.

    _send(self, job, sender)
            A routine that takes care of job submission in the case
            where self.queue == True. It does nothing in the opposite
            case as the scripts are already executed during the
            call to preoare_script in the self.queue == False case.
    """

    def __init__(self, job, sender, mode, queue=False,
                 calc_vectors=True,
                 time="00:00:00", nodes=1, ntasks=1,
                 cputask=1, memcpu=4, cd='.', name='',
                 module='Anaconda3/5.3.0', sourcename='python3imbrie'):

        super(SubmittedScript, self).__init__()

        self.modes = mode
        self.queue = queue
        self.calc_vectors = calc_vectors
        self.time = time
        self.nodes = nodes
        self.ntasks = ntasks
        self.cputask = cputask
        self.memcpu = memcpu
        self.name = name
        self.module = module
        self.sourcename = sourcename
        self.cd = cd

        self.prepare_script(job, sender)
        self._send(job, sender)

    def prepare_script(self, job, sender):
        """
        A routine for preparing scripts to
        be executed at a SLURM-based cluster
        or on the headnode/home machine.
        In case self.queue == False, the latter
        case holds. When jobs are run on the
        head node, a special distinction has
        to be made between diagonalization
        and postprocessing jobs, such as the
        sff calculation, since diagonalization
        jobs need to be performed for different
        realizations of disorder, while postprocessing
        jobs gather all diagonalization results and
        perform calculations on them in a single job.
        In the SLURM case, this is handled by job
        arrays and SLURM dependency script.

        In the self.queue == True case, a list
        of executable scripts is ultimately prepared
        which are then issued for execution using the
        self.send(*args) command.
        """

        try:
            minseed = job.params['min_seed']
            maxseed = job.params['max_seed']
        except KeyError:
            print("min_seed and max_seed keys not present!")

        #  Define commonly used objects which differ depending on
        #  whether the code is run on the SLURM-based cluster
        #  or not.

        slurm_args = [self.time, self.nodes, self.ntasks, self.cputask,
                      self.memcpu * 1000, self.name, sender.log,
                      minseed, maxseed]

        scripts = []

        # make sure that hdf5 file scripts get executed
        # following each diag job, a follow-up job is
        # performed for saving everything to hdf5
        i = 0
        j = 0
        indices = []

        modes = self.modes.copy()

        for mode in modes:

            if mode in rsc.diag_modes:
                indices.append(i + j + 1)
                j += 1

            i += 1

        for i in indices:
            modes.insert(i, 'hdf5')

        self._modes = modes.copy()
        print(self._modes)
        modes.insert(0, 'folder')
        for mode in modes:

            try:

                cmdscript = rsc.prep_sub_script(mode, self.queue,
                                                job.jobscript,
                                                sender.results,
                                                job.syspar, job.modpar,
                                                slurm_args,
                                                cmd_opt=sender.cmd_opt,
                                                slurm_opt=sender.slurm_opt,
                                                module=self.module,
                                                environment=self.sourcename,
                                                cd=self.cd,
                                                name=self.name
                                                )
                #  If jobs are submitted to the cluster,
                #  cmdscript only contains one script to be
                #  executed. This is not the case if
                #  self.queue == False and mode == 'diag',
                #  in which case multiple scripts for different
                #  seeds are prepared and then executed sequentially.

                noqueue = rsc.programs[mode]['noqueue']
                if (self.queue and not noqueue):
                    scripts.append(cmdscript[0])
                else:
                    [sp.check_call(cmd, shell=True) for cmd in cmdscript]

            except KeyError:
                print(f"Mode {mode} not yet implemented!")

            if self.queue:
                self.scripts = scripts

    def _send(self, job, sender):
        """
        Send jobs to SLURM if self.queue == True

        """
        scriptnames = []
        if self.queue:

            # prepare partial scripts
            tmp = sender.tmp
            if not os.path.isdir(tmp):
                os.makedirs(tmp)

            for mode, script in zip(self._modes, self.scripts):

                slurmscript = '{}/sbatch_{}_{}.run'.format(
                    tmp, mode, job.desc_string)
                try:
                    with open(slurmscript, 'w') as slrm:
                        slrm.write(script)
                except OSError as exc:
                    if exc.errno == 36:
                        print('Warning! Filename too long!')
                    else:
                        raise
                scriptnames.append(slurmscript)

            name = f"{tmp}/dep_script_{job.desc_string}.run"

            depscript = rsc.prepare_dependency_script(scriptnames, name)

            sp.check_call(f"sbatch {depscript}", shell=True)

        else:
            pass
