from postprocessing import disorder as dis
from postprocessing import data_extraction as dae

from entro_analyse import PathSetter

topdir = ('/scratch/jan/qmbmodels/results/'
          'spin1d_xxz_get_entropy_si_target_ave_ener')

paramsIso = (
    topdir,
    ('/home/jan/'
     'XXZ_isotropic_entro_post_nev_100_n_dependence'),
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
     'XXZ_J1_J2_ergodic_post_nev_100_n_dependence'),
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

        kwargs_dict = {
            'target_variance': model.var_prefactor,
            'population_variance': True,
        }
        savepath = model.savepath
        dae.extract_data(topdir, savepath, routine='entro_analyse',
                         partial=True, disorder_key='dW',
                         savename='entro_n_dependnce', reverse_order=True,
                         exclude_keys=model.exclude_keys,
                         collapse=True, merge=False,
                         **kwargs_dict)
