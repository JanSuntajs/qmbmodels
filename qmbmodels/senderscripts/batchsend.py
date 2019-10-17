"""
A module with an implementation of a class
that takes care of preparation of a set(batch)
of jobs to be executed on a SLURM-based
cluster or in parallel on the headnode/home
machine.

"""

import os
import itertools as it

from .prepjob import Job
from .scriptsub import SubmittedScript


class BatchSender(object):

    """
    A job for preparing and sending a batch
    of jobs.

    Parameters:

    params: dict
        A dictionary of parameter string value
        lists. Example:

        params = {
            'L': [8, 10],
            'J': [0.0],
            'dJ': [1.0],
            'W': [0.0],
            'dW': [1.0],
            'Gamma': [0.],
            'dGamma': [1.0],
            'min_seed': [1],
            'max_seed': [10],
            }

    syspar_keys: list
        A list of strings which specifies
        which parameters correspond to the
        system, such as system size or
        dimensionality. Example:
        syspar_keys = syspar_keys = ['L']

    modpar_keys: list
        A list of strings which specifies
        which parameters correspond to the
        Hamiltonian modules. Example:

        modpar_keys = ['J', 'dJ', 'W', 'dW', 'Gamma',
                       'dGamma', 'min_seed', 'max_seed']

    storage: string, path
        Path to the root directory of
        where the programs should be
        ran from and where also the
        results and log files are
        stored.

    Attributes:
        params, _syspar_keys, _modpar_keys:
        see the above "Parameters" section.

    Routines:
        prepare_jobs(self):
        a routine for preparing a list of jobs
        to be executed.

        run_jobs(self):
        a routine for running the jobs from
        the self.jobs list.


    """

    def __init__(self, params, syspar_keys, modpar_keys, auxpar_keys=[],
                 storage='.'):
        super(BatchSender, self).__init__()

        self.params = params
        self._syspar_keys = syspar_keys
        self._modpar_keys = modpar_keys
        self._auxpar_keys = auxpar_keys
        self.prepare_jobs()
        self.prepare_folders(storage)

    def __repr__(self):

        return f"A batch of jobs to be executed."

    def prepare_folders(self, storage):
        """
        A function that prepares the
        relevant folder structure for
        running the scripts and saving
        the data and outputing log files.

        """

        self.storage = storage
        subdirs_list = ['tmp', 'log', 'results']
        subdirs_dict = {}
        # make the storage directory if it
        # is not yet present
        if not os.path.isdir(storage):
            os.mkdir(storage)

        for subdir in subdirs_list:

            sub_path = self.storage + '/' + subdir + '/'
            subdirs_dict[subdir] = sub_path
            if not os.path.isdir(sub_path):
                os.mkdir(sub_path)
        self.__dict__.update(subdirs_dict)

    def prepare_jobs(self):
        """
        A function using the itertools.product
        function to package different combinations
        of system and module parameters into a
        list of jobs -> the functions accepts
        a dictionary of parameter value lists
        and returns a list of Job class instances.

        Parameters:

        params: dict
            A dictionary of parameter string value
            lists. Example:

            params = {
                'L': [8, 10],
                'J': [0.0],
                'dJ': [1.0],
                'W': [0.0],
                'dW': [1.0],
                'Gamma': [0.],
                'dGamma': [1.0],
                'min_seed': [1],
                'max_seed': [10],
                }
        """

        keys = self.params.keys()
        vals = self.params.values()
        jobs = []

        for i in it.product(*vals):

            add_items = dict(zip(keys, i))

            job = Job(add_items, self._syspar_keys,
                      self._modpar_keys, self._auxpar_keys)

            jobs.append(job)

        self.jobs = jobs

    def run_jobs(self, mode, queue=False, time="00:00:01", nodes=1,
                 ntasks=1, cputask=1, memcpu=4, module='Anaconda3/5.3.0',
                 sourcename='python3imbrie', name=''):
        """
        A function for running jobs either on the headnode/home machine or in
        the SLURM based cluster.

        Parameters:

        queue: boolean, optional
            If True, the code is executed on a SLURM-based cluster. Defaults
            to False
        time: string, optional
            Format: "DD-HH:MM:SS" (days-hours:minutes:seconds). Determines
            the expected runtime of the program on SLURM.
        nodes: int, optional
            The number of nodes to be used for the calculation. Defaults to 1.
        nthreads: int, optional
            The number of threads to use in calculation - how many CPUS are
            reserved for the calcualation in the SLURM case.
        memcpu: int
            memory per cpu core (thread) in GB.
        """

        for job in self.jobs:

            script = SubmittedScript(
                job, self, mode, queue=queue, time=time, nodes=nodes,
                ntasks=ntasks, cputask=cputask, memcpu=memcpu,
                module=module, sourcename=sourcename, name=name)
