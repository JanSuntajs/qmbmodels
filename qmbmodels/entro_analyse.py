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
          'spin1d_xxz_get_entropy_si_target_ave_ener')

paramsIso = (
    topdir,
    ('/home/jan/'
     'XXZ_isotropic_entro_post_nev_100_epsilon_{:.3f}/mode_{}'),
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
     'XXZ_J1_J2_ergodic_post_nev_100_epsilon_{:.3f}/mode_{}'),
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

    for model in [pathsErgod, pathsIso]:

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
        epsilon_theor = dW_min**2 * model.var_prefactor - 0.1
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
            dae.extract_data(topdir, savepath, routine='get_r',
                             partial=True, disorder_key='dW',
                             savename='r_sweep_post', reverse_order=True,
                             exclude_keys=model.exclude_keys,
                             collapse=True,
                             **kwargs_dict)
