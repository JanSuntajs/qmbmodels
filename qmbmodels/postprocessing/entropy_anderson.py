"""
This module implements routines for analysing the
entanglement entropy results.

"""


import numpy as np

from .disorder import _preparation, _preparation_analysis


footer_entro_and = """
Each row is organised as follows:

0) dW: disorder strength

1) average_entropy S_ave: average entropy for a given number of states and
   samples

2) theoretical entropy S_th:
   (1 - (1 + f^(-1)*(1-f)*ln(1-f))/ln(2))*VA*ln(2);
   V-> system,
   VA-> subsystem

3) S_re: rescaled entropy: (S_th - S_ave) / (VA*ln(2))

4) Delta S_ave: standard deviation of S_ave (not rescaled, both sample
   to sample and state to state contributions)

5) size L: system size

6) fraction: subsystem size fraction compared to the whole system

7) nener: number of energies obtained using partial diagonalization

8) target_variance: Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2.
    If mode=0: nan, this argument is not needed if preprocessing is not
    performed.

9) epsilon: condition used to determine whether to select a given disorder
   distribution.
   If mode=0: nan

10) dW_min: value of the disorder strength parameter for which the epsilon
   was evaluated.
   If mode=0: nan

11) variance_before: variance of variances of the disorder distributions before
    post.

12) variance_after: variance of variances of the disorder distributions after
    post.

13) nsamples: number of all the random samples

14) nsamples_selected: number of the random disorder samples with an
    appropriate variance.
    NOTE: if mode (entry 15) ) equals 0, nsamples equals nsamples.

15) nsamples_rejected: nsamples - nsamples_selected
    NOTE: if mode (entry 15) ) equals 0, this should be equal to 0.

16) mode: which postprocessing mode was selected
    NOTE: 0 indicates no postprocessing!

17) population_variance: integer specifying whether theoretical prediction for
    the population variance was used in order calculate the target variance.
    1 if that is the case, 0 if not.
"""


def _entro_ave(entropy, condition, size, sample_averaging=True,
               *args, **kwargs):

    # theoretical results
    fraction = kwargs['filling_fraction']
    subsize = size * fraction
    # prefactor terms
    fminus = 1 - fraction
    term1 = 1.
    term2 = 1 + fminus * np.log(fminus) / (fraction)
    term2 *= 1 / np.log(2)

    entro_theory = (term1 - term2) * subsize * np.log(2)

    ave_entro = np.mean(entropy)
    # sample fluctuations - across samples and states
    std_entro = np.std(entropy)
    entro_rescaled = (entro_theory - ave_entro) / (subsize * np.log(2))

    output = (ave_entro, entro_theory, entro_rescaled,
              std_entro, size, fraction)

    output = tuple(map(np.atleast_1d, output))
    return output


def entro_ave_and(h5file,
                  results_key='Entanglement_entropy_Anderson_filling_fraction',
                  disorder_key='dW',
                  target_variance=1. / 3.,
                  epsilon=0.,
                  dW_min=0.,
                  population_variance=True,
                  mode=0,
                  disorder_string='Hamiltonian_random_disorder_partial',
                  sample_averaging=True,
                  *args,
                  **kwargs):
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

    ave_entro: float
    Average entropy of the eigenstates in all the available
    disorder realizations. Note that currently we only implement
    the calculation for a symmetric bipartition (sub = size / 2).

    entro_rescaled: float
    Rescaled value of the entanglement entropy, where the rescaling
    is as follows:
    entro_rescaled = |log(2) - 2**(2 * sub - size - 1) / sub -
                                  ave_entro / sub|
    here, sub is the subsystem size.

    std_entro: float
    Standard deviation of the calculated entropies.

    std_entro_rescaled: float
    Standard deviation of the rescaled entropies.

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

    results = _preparation(h5file, results_key, disorder_key,
                           disorder_string,
                           target_variance, population_variance,
                           mode, epsilon, dW_min,
                           _entro_ave, 6,
                           sample_averaging=sample_averaging,
                           *args,
                           **kwargs)
    return results
