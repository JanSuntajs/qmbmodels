"""
Implementation of a simple class used in data
extraction routines which are meant for
performing the data analysis and parameter
sweeps.

"""


class PathSetter(object):

    def __init__(self, topdir, savepath, descriptor,
                 syspar_min, min_dis, modpar_min,
                 var_prefactor, exclude_keys):
        super(PathSetter, self).__init__()

        self.topdir = topdir
        self.descriptor = descriptor
        self.syspar_min = syspar_min
        self.min_dis = min_dis
        self.modpar_min = modpar_min.format(self.min_dis)
        self.var_prefactor = var_prefactor
        self.exclude_keys = exclude_keys
        self.savepath = savepath