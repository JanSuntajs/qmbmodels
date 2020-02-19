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


topdir = ('/scratch/jan/qmbmodels/results/'
          'spin1d_xxz_get_entropy_si_target_ave_ener_nev_500')

paramsIso = (
    topdir,
    ('/home/jan/'
     'XXZ_isotropic_entro_post_nev_500_epsilon_{:.3f}/mode_{}'),
    'pbc_True_disorder_uniform_ham_type_spin1d',
    'L_12_nu_6',
    1.0,
    ('J1_1.0_J2_0.0_delta1'
     '_1.0_delta2_0.0_W_0.0_dW_{}'),
    1. / 3.,
    ['J1_1.0_J2_1.0_delta1_0.55_delta2_0.55_W_0.0', 'L_12_nu_8'],
)


paramsErgod = (
    topdir,
    ('/home/jan/'
     'XXZ_J1_J2_ergodic_post_nev_500_epsilon_{:.3f}/mode_{}'),
    'pbc_True_disorder_uniform_ham_type_spin1d',
    'L_12_nu_6',
    2.0,
    ('J1_1.0_J2_1.0_delta1'
     '_0.55_delta2_0.55_W_0.0_dW_{}'),
    1. / 3.,
    ['J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0', 'L_12_nu_8'],
)


pathsIso = PathSetter(*paramsIso)
pathsErgod = PathSetter(*paramsErgod)


if __name__ == '__main__':

    for model in [pathsErgod]:

        (dW_min, means, variances,
         std_variances,
         rescale_factor) = dis.get_min_variance(model.topdir, model.descriptor,
                                                model.syspar_min,
                                                model.modpar_min, 'dW')

        # if we calculate the minimum stopping condition
        # from the numerics
        epsilon_num = std_variances * rescale_factor
        # we can also do it theoretically -> from the prediction
        # for the minimum std of the variances, which is, for a
        # box distribution, approximately equal to dW_min**2/3
        # this is \Sigma_0(W_min) from our notes
        epsilon_theor = dW_min**2 * model.var_prefactor
        # epsilon_theor = 0.8
        # epsilon_theor = 0.2
        for mode in [0, 1, 2]:
            kwargs_dict = {
                'target_variance': model.var_prefactor,
                'epsilon': epsilon_theor,
                'population_variance': True,
                'mode': mode,
                'dW_min': model.min_dis,
            }
            savepath_ = model.savepath
            savepath = savepath_.format(epsilon_theor, mode)
            dae.extract_data(topdir, savepath, routine='get_entro_ave',
                             partial=True, disorder_key='dW',
                             savename='entro_sweep_post', reverse_order=True,
                             exclude_keys=model.exclude_keys,
                             collapse=True,
                             **kwargs_dict)
