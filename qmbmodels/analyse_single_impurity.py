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


# topdir = ('/scratch/jan/qmbmodels/results/'
#           'heisenberg_single_impurity_spin1d_test')
topdir = ('./results/'
          'heisenberg_single_impurity_spin1d_check_degs_special_points')

paramsAnder = (
    topdir,
    ('./results/'
     'heisenberg_single_impurity_spin1d_check_degs_special_points_post'),
    'pbc_False_disorder_single_ham_type_anderson',
    'L_12_nu_6',
    1.0,
    'J_4.0_dJ_0.0_delta_0.5_ddelta_0.05_W_0.0_dW_{:.5f}_noise_0.05000',
    1. / 3.,
    [],
)


pathsAnder = PathSetter(*paramsAnder)


if __name__ == '__main__':

  for model in [pathsAnder]:

    dW_min = model.min_dis
    # if we calculate the minimum stopping condition
    # from the numerics
    # we can also do it theoretically -> from the prediction
    # for the minimum std of the variances, which is, for a
    # box distribution, approximately equal to dW_min**2/3
    # this is \Sigma_0(W_min) from our notes
    epsilon_theor = dW_min**2 * model.var_prefactor - 0.1
    # epsilon_theor = 0.8
    # epsilon_theor = 0.2

    methods = [['get_deg', 'deg_sweep_post'],]
               #['get_r', 'r_sweep_post'],
               #'get_entro_ave', 'entro_sweep_post']]

    kwargs_dict = {
        'target_variance': model.var_prefactor,
        'epsilon': epsilon_theor,
        'population_variance': True,
        'mode': 0,
        'dW_min': model.min_dis,
    }
    savepath_ = model.savepath

    for method in methods:
      dae.extract_data(topdir, savepath_, routine=method[0],
                       partial=False, disorder_key='delta',
                       disorder_string='Hamiltonian_random_disorder',
                       savename=method[1], reverse_order=True,
                       exclude_keys=model.exclude_keys,
                       collapse=True,
                       **kwargs_dict)
