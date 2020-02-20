
from .disorder import _preparation


footer_r_ave = """
Each row is organised as follows:

0) dW: disorder strength

1) <r>: the mean ration of the adjacent level spacings where the
   means < \tilde r> are first obtained for individual spectra
   and are then averaged over different disorder realizations.

2) delta <r>: standard deviation of the distribution of r values
   for different disorder realizations.

3) size L: system size

4) nener: number of energies obtained using partial diagonalization

5) target_variance: Variance of the disordered samples
with which to compare the numerical results in the postprocessing
steps if mode equals 1 or 2. Defaults to None as the argument is not
required in the mode=0 case where postprocessing is not performed.

6) epsilon: condition used to determine whether to select a given disorder
distribution.
If mode=1:
If mode=2:

7) dW_min: value of the disorder strength parameter for which the epsilon
was evaluated.

8) variance_before: variance of variances of the disorder distributions before
post.

9) variance_after: variance of variances of the disorder distributions after
post.

10) nsamples: number of all the random samples

11) nsamples_selected: number of the random disorder samples with an
appropriate variance.

12) nsamples_rejected: nsamples - nsamples_selected

13) mode: which postprocessing mode was selected

14) population_variance: integer specifying whether theoretical prediction for
the population variance was used in order calculate the target variance.
1 if that is the case, 0 if not.
"""


def _r_ave(rvals, condition, size, *args, **kwargs):

    r_val = rvals[0][1]
    r_err = rvals[0][2]

    return r_val, r_err


def r_ave(h5file, results_key='r_data',
          disorder_key='dW',
          target_variance=1. / 3.,
          epsilon=0.,
          dW_min=0.,
          population_variance=True,
          mode=0,
          disorder_string='Hamiltonian_random_disorder_partial',
          *args,
          **kwargs
          ):
    """
    A routine for performing statistical analysis of the entanglement
    entropy results which also allows for performing the variance
    reduction postprocessing steps.

    Parameters:
    -----------

    h5file: string
    Filename of the .hdf5 file containing the numerical data to be
    analysed.

    target_variance: {float, None}
    Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2. Defaults to None as the argument is not
    required in the mode=0 case where postprocessing is not performed.
    See also the explanation for the population_variance parameter.

    epsilon: {float, None}
    Stopping criterion for the postprocessing variance reduction routines
    if mode=1 or mode=2. In the absence of postprocessing, epsilon is set
    to None.

    dW_min: {float, None}
    the disorder strength parameter value for which the epsilon
    was evaluated. Set to None in the absence of postprocessing.

    population_variance: boolean
    If True, target variance is the population variance of the
    disorder distribution. In that case, the value of the target_variance
    argument should be a numerical prefactor to the square of the
    disorder parameter's strength: target_variance =
    dW**2 * target_variance
    If False, target_variance is calculated as the mean of the distribution
    of the sample variances.

    results_key: string, optional
    String specifying which results to load from the .hdf5 file.
    The argument
    is provided for compatibility of future versions of this code package
    in case the key under which the entropy results are stored in the .hdf5
    files might change in some manner.

    disorder_key: string, optional
    String specifying which parameter descriptor describes disorder.

    mode: int, optional
    Which of the postprocessing modes to use:
            0: no postprocessing
            1: first variance reduction scheme
            2: second variance reduction scheme
    See the documentation of the disorder.reduce_variance(...) function
    for further details.

    disorder_string: string, optional
    String specifying the key under which the disordered spectra are
    stored in thethe .hdf5 file. The argument
    is provided for compatibility of future versions of this code package
    in case the key under which the disorder samples are stored in the
    .hdf5 files might change in some manner.

    Returns:
    --------

    dW: float
    Value of the disorder strength parameter

    r_mean: float
    Mean ratio of the adjacent level spacings.

    r_std: float
    Standard deviation of the <r> values calculated
    for different random spectra.

    size, nener: int
    System size and number of energies in the spectra, respectively.

    target_variance, epsilon: see above
    in the Parameters section.

    std_before, std_after: float
    Standard deviation of variances of the disordered samples before
    and after the postprocessing procedure.

    nsamples, nsamples_selected, nsamples_rejected: int
    Number of all the available samples (disorder realizations),
    as well as the number of the rejected and selected ones in the
    case of a postprocessing step.

    mode: int
    Which postprocessing mode was selected.

    population_variance: int
    Whether theoretical population variance was used to
    estimate the target variance or not.
    """

    return _preparation(h5file, results_key, disorder_key,
                        disorder_string,
                        target_variance, population_variance,
                        mode, epsilon, dW_min,
                        _r_ave, 2, *args, **kwargs)
