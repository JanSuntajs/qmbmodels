#!/usr/bin/env python
"""
This routine prepares the folder structure for
the file


"""

import os

from cmd_parser_tools import arg_parser


if __name__ == '__main__':

    # argsDict -> system and module dependent parameters
    # extra -> path for saving the results
    argsDict, extra = arg_parser([], [])

    savepath = argsDict['results']
    syspar = argsDict['syspar']
    modpar = argsDict['modpar']

    metapath = os.path.join(savepath, 'metadata')
    # ----------------------------------------------------------------------

    if not os.path.isdir(savepath):

        os.makedirs(savepath, exist_ok=True)

    if not os.path.isdir(metapath):

        os.makedirs(metapath, exist_ok=True)
