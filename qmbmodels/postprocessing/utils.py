import numpy as np
import os


def _extract_disorder(string, disorder_key):
    """
    An internal routine for returning the
    disorder key and the rest of the input
    string

    Parameters:
    -----------

    string: string
    A string from which the disorder key and its
    corresponding value are to be extracted. Example:
    'J1_1.0_J2_1.0_delta1_0.55_delta2_0.55_W_0.0_dW_1'

    disorder_key: string
    String designating which parameter descriptor
    corresponds to the disorder strength parameter.
    Example: 'dW'

    Returns:
    --------

    rest: string
    Part of the filename without the
    disorder key and its value. For the above example
    string and disorder_key, that would be:
    'J1_1.0_J2_1.0_delta1_0.55_delta2_0.55_W_0.0'

    disorder: float
    The numerical value of the disorder corresponding
    to the disorder key. In the above case, the return
    would be:
    1.0

    """

    # append '_' at the beginning of the
    # string to make splitting w.r.t. the
    # disorder_key easier
    string = '_' + string

    # make sure there are no trailing or preceeding
    # multiple underscore by removing them
    disorder_key = disorder_key.lstrip('_').rstrip('_')
    # now make sure there is exactly one trailing
    # and one preceeding underscore
    disorder_key = '_{}_'.format(disorder_key)
    # split w.r.t. the disorder_key. The first
    # part does not contain the disorder parameter
    # value, while the second one does
    rest1, dis_string = string.split(disorder_key)
    # find the first occurence of '_' in the
    # dis_string, which indicates the length of
    # the disorder parameter value
    splitter = dis_string.find('_')
    if splitter < 0:
        disorder = dis_string
        rest2 = ''
    else:
        disorder, rest2 = dis_string[:splitter], dis_string[splitter:]

    disorder = np.float(disorder)

    # the part without the disorder value
    rest = rest1.lstrip('_') + rest2
    return rest, disorder


def extract_single_model(topdir, descriptor, syspar, modpar):
    """
        A function for extracting the location of a file containing
        the numerical results for a single/unique set of data
        parameters

        The entries topdir, descriptor, syspar and modpar are strings
        from which the full path to the requested file is constructed.

    """
    filepath = os.path.join(topdir, descriptor, syspar, modpar)
    if os.path.isdir(filepath):

        try:

            file = glob('{}/*.hdf5'.format(filepath))[0]

        except IndexError:
            print('file in folder {} not present!'.format(filepath))

    else:

        print('folder {} does not exist!'.format(filepath))

        file = None

    return file
