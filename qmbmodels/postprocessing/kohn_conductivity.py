
"""
Code for analysing gamma

"""


import numpy as np

from .disorder import _preparation


footer_kohn_ave = """
Each row is organised as follows:

0) dW: disorder strength

1) \phi: phase

2) nener0: number of all energies in the spectra

3) nener: number of energies used in the calculations

4) g1: kohn conductivity calculated by taking the
   ratio of Thouless and Heisenberg energy for each disorder
   sample, then averaging the results over disorders

5) g2: kohn conductivity obtained by taking the global
   means for thouless and heisenberg energy, then taking
   their ratio

6) E_T: Thouless energy, mean over all spectra and disorder
   realizations

7) E_H: Heisenberg energy, mean level spacing over all
   spectra and disorder realizations

8) standard deviation of g1

9) standard deviation of E_T

10) standard deviation of E_H

11) size L: system size

12) nener: number of energies obtained using partial diagonalization

13) target_variance: Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2.
    If mode=0: nan, this argument is not needed if preprocessing is not
    performed.

14) epsilon: condition used to determine whether to select a given disorder
   distribution.
   If mode=0: nan

15) dW_min: value of the disorder strength parameter for which the epsilon
   was evaluated.
   If mode=0: nan

16)  variance_before: variance of variances of the disorder distributions before
    post.

17)  variance_after: variance of variances of the disorder distributions after
    post.

18) nsamples: number of all the random samples

19) nsamples_selected: number of the random disorder samples with an
    appropriate variance.
    NOTE: if mode (entry 15) ) equals 0, nsamples equals nsamples.

20) nsamples_rejected: nsamples - nsamples_selected
    NOTE: if mode (entry 15) ) equals 0, this should be equal to 0.

21) mode: which postprocessing mode was selected
    NOTE: 0 indicates no postprocessing!

22) population_variance: integer specifying whether theoretical prediction for
    the population variance was used in order calculate the target variance.
    1 if that is the case, 0 if not.
"""


def _kohn_ave(kohnvals, condition, size, *args, **kwargs):

    kohn_vals = kohnvals[0]

    return tuple(map(np.atleast_1d, kohn_vals))


def kohn_ave(h5file, results_key='kohn_data',
             disorder_key='dW',
             target_variance=1. / 3.,
             epsilon=0.,
             dW_min=0.,
             population_variance=True,
             mode=0,
             disorder_string='Hamiltonian_random_disorder',
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
                        _kohn_ave, 10, *args, **kwargs)
