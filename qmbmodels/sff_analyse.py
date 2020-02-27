"""
This main function contains tools and routines
for performing a statistical analysis of the
entanglement entropy calculations w.r.t the
width of their corresponding disorder distributions.

TO DO:
Move the PathSetter class to a separate module
in the postprocessing subpackage.

Currently, we handle two different sets of results,
namely, for the isotropic and XXZ case with included
next-nearest neighbour contributions.

Attributes:
-----------

topdir: str
        Path to the topfolder storing the numerical
        results.

paramsISO, paramsErgod: tuple
        Tuples containing the data needed to instantiate
        instances of the PathSetter class for the
        isotropic and ergodic (XXZ) case, respectively.
        The contents of the tuple should be:
        0) <topdir> root folder of the numerical data
           storage location
        1) <savepath> e.g. where to store the postprocessed
           results
        2) <syspar> system parameters of the system we use
           in order to numerically obtain the standard
           deviation of the disorder distribution. For
           example: <syspar> = 'L_12_nu_6'
        3) <disorder> value of the disorder for which
           we wish to determine the epsilon criterion
           for the disorder distribution. Example:
            <disorder> = 1.0
        4) <min_modpar_template> string giving the
           name of the folder in which the data for the
           calculation of the epsilon criterion are
           stored. Example:
           ('J1_1.0_J2_0.0_delta1'
            '_1.0_delta2_0.0_W_0.0_dW_{}')
        5) <var_prefactor> the value of the
           multiplicative prefactor with which we
           need to multiply the square of the
           disorder parameter in order to obtain the
           theoretical prediction for the disorder
           variance.
        6) <exclude folders> a list of (sub)folders
           not to visit during the folder crawl.

"""


from postprocessing import disorder as dis
from postprocessing import data_extraction as dae

from entro_analyse import PathSetter

topdir = ('/scratch/jan/MBLexact_py/results/'
          'to_py_from_fortran')

paramsIso = (
    topdir,
    ('/home/jan/'
     'XXZ_isotropic_sff_analysis/'),
    'pbc_True_disorder_uniform_ham_type_spin1d',
    'L_12_nu_6',
    1.0,
    ('J1_2.0_J2_0.0_delta1'
     '_1.0_delta2_0.0_W_0.0_dW_{}'),
    1. / 3.,
    ['J1_-2.0_J2_-2.0_delta1_-0.55_delta2_-0.55_W_0.0', 'L_12_nu_8'],
)


paramsErgod = (
    topdir,
    ('/home/jan/'
     'XXZ_J1_J2_ergodic_sff_analysis/'),
    'pbc_True_disorder_uniform_ham_type_spin1d',
    'L_12_nu_6',
    2.0,
    ('J1_2.0_J2_2.0_delta1'
     '_-0.55_delta2_-0.55_W_0.0_dW_{}'),
    1. / 3.,
    ['J1_2.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0', 'L_12_nu_8'],
)


pathsIso = PathSetter(*paramsIso)
pathsErgod = PathSetter(*paramsErgod)


if __name__ == '__main__':

    for model in [pathsErgod, pathsIso]:

        method = 'get_sff'
        for mode in [0]:
            kwargs_dict = {
                'population_variance': True,
                'mode': mode,
            }
            savepath_ = model.savepath
            savepath = savepath_.format(mode)


            dae.extract_data(topdir, savepath, routine=method,
                             partial=True, disorder_key='dW',
                             savename=method[1], reverse_order=True,
                             exclude_keys=model.exclude_keys,
                             collapse=True,
                             **kwargs_dict)
