"""
A module implementing the class
for creating a Job -> an object
that stores all the relevant
information, such as the
system and model parameters,
about the actual physical hamiltonian.
The class also prepares formatted
strings that provide job descriptions
and command line arguments to be
given to the executable scripts.

"""


# default syspar and modpar keys for the Imbrie model.
# this can be easily generalized to other models.
# syspar_keys = ['L']
# modpar_keys = ['J', 'dJ', 'W', 'dW', 'Gamma', 'dGamma', 'min_seed',
#                'max_seed']

#selected_keys = syspar_keys + modpar_keys


class Job(object):

    """
    A class that stores the model
    and system parameters and
    prepares a single job for running.

    Parameters:

    params: dict
        A dictionary of parameter (string) values
        with their corresponding keys. Example

        params = {
            'L': 8,
            'J': 0.0,
            'dJ': 1.0,
            'W': 0.0,
            'dW': 1.0,
            'Gamma': 0.,
            'dGamma': 1.0,
            'min_seed': 1,
            'max_seed': 10,
            }

    redef: dict
           Dictionary with redefinitions of params dict entries.
           This is especially useful when one wishes params dict
           entries to depend on other params dict entries. An example
           redef entry where we wish one entry to depend on another
           one would be written as follows:

           redef = {'max_seed': lambda x: {'max_seed': x['min_seed'] + 10}}

           The required structure is thus as follows: the value corresponding
           to a key should be a lambda expression (callable) returning a dict
           with a matching key and specifying how the redefined value
           should depend on other dictionary values.

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

    auxpar_keys: list
        A list of strings which specifies which
        parameters correspond to the auxiliary
        parameters, such as the parameters in the
        postprocessing steps (for instance,
        in sff or spectral statistics calculations).
        Example:

        auxpar_keys = ['min_tau', 'max_tau', 'n_tau',
                       'unfolding_n', 'sff_filter']

    Attributes:

        params, _syspar_keys, _modpar_keys, auxpar_keys: see above in the
        "Parameters" section.
        selected_keys: list
            A combined list of syspar_keys and modpar_keys.
            The rule is simply:
            self.selected_keys = self._syspar_keys + self.modpar_keys

        jobscript: string
            A string which is added to the executable script at
            runtime and provides values of system and model
            parameters.
            Example:
            jobscript = '--L=8 --J=0.0 --dJ=1.0 --W=0.0 --dW=1.0 --Gamma=0.0
            --dGamma=1.0 --min_seed=1 --max_seed=10'

        jobdesc: string
            A string which describes the job parameters.
    Routines:

        prepare_job(self)
        A routine for preparing the jobscript string.

    Routines:
    """

    def __init__(self, params, syspar_keys, modpar_keys, auxpar_keys=[],
                 redef={}):
        super(Job, self).__init__()
        self.params = params
        self._redef = redef
        self._update_params()
        self._syspar_keys = syspar_keys
        self._modpar_keys = modpar_keys
        self._auxpar_keys = auxpar_keys
        self.selected_keys = self._syspar_keys + self._modpar_keys \
            + self._auxpar_keys
        self.prepare_job()

    def __repr__(self):

        return f"A job to be executed."

    def _update_params(self):
        """
        A function for modifying keys and values
        of the params dictionary.
        """

        # copy the initial dict of parameters
        parsed_params = self.params.copy()

        for key in self._redef.keys():

            parsed_params.update(self._redef[key](self.params))

        self.params = parsed_params

    def prepare_job(self):
        """
        A function which prepares
        jobscript. -> the functions accepts
        a dictionary of parameter keys and values
        and generates a string.

        Parameters:

        params: dict
            A dictionary of parameter string value
            lists. Example:

            params = {
                'L': 8,
                'J': 0.0,
                'dJ': 1.0,
                'W': 0.0,
                'dW': 1.0,
                'Gamma': 0.,
                'dGamma': 1.0,
                'min_seed': 1,
                'max_seed': 10,
                }

        Returns:

        jobscript: list
            A list of formatted strings which are provided
            as commandline arguments to the runtime scripts.
            Example for the above case of parameters:

            '--L=8 --J=0.0 --dJ=1.0 --W=0.0 --dW=1.0 --Gamma=0.0 --dGamma=1.0
             --min_seed=1 --max_seed=10'

        """
        command_script = ""
        desc_string = ""
        syspar = ""
        modpar = ""
        auxpar = ""

        modpar_keys = [k for k in self._modpar_keys if 'seed'
                       not in k]

        for key in self._syspar_keys:
            syspar += "{}_{{}}_".format(key)

        for key in modpar_keys:

            modpar += "{}_{{}}_".format(key)

        for key in self._auxpar_keys:

            auxpar += "{}_{{}}_".format(key)

        for key in self.selected_keys:
            command_script += "--{}={{}} ".format(key)
            desc_string += "{}_{{}}_".format(key)

        def _make_jobstrings(template, keys):
            """
            A function that prepares a string describing
            a job given a template according to which
            the string should be formatted and a list
            of keys from some dictionary which should
            specify key and value pairs.

            Parameters:

            template: string
                A string formatting pattern, such as the
                following example:
                template = "--J={} --dJ={}" (commandline script)
                template = "J_{}_dJ_{}" (job desc. string)
            keys: list
                A list of dictionary keys.
            """

            formatted_template = template.strip("_").strip(" ").format(
                *[self.params[key] for key in keys])

            return formatted_template

        jobscript, desc_string = [_make_jobstrings(template,
                                                   self.selected_keys)
                                  for template in
                                  [command_script, desc_string]]
        syspar, modpar, auxpar = [_make_jobstrings(*args)
                                  for args in
                                  [(syspar, self._syspar_keys),
                                   (modpar, modpar_keys),
                                   (auxpar, self._auxpar_keys)]]

        self.jobscript = jobscript
        self.desc_string = desc_string
        self.syspar = syspar
        self.modpar = modpar
        self.auxpar = auxpar
