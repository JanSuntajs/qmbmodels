# import numpy as np

from postprocessing import disorder as dis
# from postprocessing import entropy as ent
from postprocessing import data_extraction as dae


topdir = ('/scratch/jan/qmbmobels/results/'
          'spin1d_xxz_get_entropy_si_target_ave_ener')
savepath = ('/home/jan/'
            'Heisenberg_model_test_analysis/')
descriptor = 'pbc_True_disorder_uniform_ham_type_spin1d'
syspar = 'L_12_nu_6'
modpar = 'J1_1.0_J2_0.0_delta1_1.0_delta2_0.0_W_0.0_dW_1.0'
# prefactor for variances depending on the chosen distribution
var_prefactor = 1. / 3.

if __name__ == '__main__':

    # get target_deviation for an actual spectrum
    (dW_min, means, variances,
     std_variances,
     rescale_factor) = dis.get_min_variance(topdir, descriptor,
                                            syspar, modpar, 'dW')

    # if we calculate the minimum stopping condition
    # from the numerics
    epsilon_num = std_variances * rescale_factor
    # we can also do it theoretically -> from the prediction
    # for the minimum std of the variances, which is, for a
    # box distribution, approximately equal to dW_min**2/3
    # this is \Sigma_0(W_min) from our notes
    epsilon_theor = dW_min**2 * var_prefactor

    kwargs_dict = {
        'target_variance': 1. / 3.,
        'epsilon': epsilon_theor,
        'population_variance': True,
        'mode': 0,
    }

    dae.extract_data(topdir, savepath, routine='get_entro_ave',
                     partial=True, disorder_key='dW',
                     savename='entro_sweep_post', reverse_order=True,
                     **kwargs_dict)
