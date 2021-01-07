"""
This module implements routines for analysing the
Thouless conductivity of the Anderson model as
implemented in: 

https://iopscience.iop.org/article/10.1088/0022-3719/5/8/007
"""


import numpy as np
from scipy.stats.mstats import gmean
from math import ceil, floor
from .disorder import _preparation, _preparation_analysis


footer_thoucond_ave = """
Each row is organised as follows:

0) dW: disorder strength

1) ari_ari_mean: arithmetic mean across both
   samples and energies where entry 9) refers
   to the number of energies considered in the analysis.

2) ari_geo_mean: geometric mean of the energies in a single
   disorder sample and the arithmetic mean of the means
   of individual samples.

3) geo_ari_mean: first, an arithmetic mean of the energies
   in a single sample is taken and then the geometric mean
   of the single sample mean values.

4) geo_geo_mean: geometric mean of all the energies across
   individual spectra and given disorder realizations.

5) thouless conductivity g, obtained by 
   dividing the thouless energy by the heisenberg energy -> 
   multiplying 1) by 10)

6) analogous to 5), obtained by multiplying 2) by 10)
 
7) analogous to 4) and 5), obtained by multiplying 3) by 10)

8) ... obtained by multiplying 4) by 10)

9) nener: number of the energies considered in the calculations

10) hilbert_dim: the full hilbert space dimension

11) Delta_thou_cond: standard deviation of the thouless conductivity

12) size L: system size

13) disregard this entry completely

14) target_variance: Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2.
    If mode=0: nan, this argument is not needed if preprocessing is not
    performed.

15) epsilon: condition used to determine whether to select a given disorder
   distribution.
   If mode=0: nan

16) dW_min: value of the disorder strength parameter for which the epsilon
   was evaluated.
   If mode=0: nan

17) variance_before: variance of variances of the disorder distributions before
    post.

18) variance_after: variance of variances of the disorder distributions after
    post.

19) nsamples: number of all the random samples

20) nsamples_selected: number of the random disorder samples with an
    appropriate variance.
    NOTE: if mode (entry 15) ) equals 0, nsamples equals nsamples.

21) nsamples_rejected: nsamples - nsamples_selected
    NOTE: if mode (entry 15) ) equals 0, this should be equal to 0.

22) mode: which postprocessing mode was selected
    NOTE: 0 indicates no postprocessing!

23) population_variance: integer specifying whether theoretical prediction for
    the population variance was used in order calculate the target variance.
    1 if that is the case, 0 if not.
"""


def _thoucond_ave(thoucond, condition, size, sample_averaging=True,
                  nener=-1,
                  *args, **kwargs):

    # nener: number of energies to consider; if -1, the whole spectrum
    # is considered

    thoucond = np.abs(thoucond)
    nsamples, hilbert_dim = thoucond.shape

    if ((nener == -1) or (hilbert_dim > nener)):
        nener = hilbert_dim
    else:
        med_ener = int(hilbert_dim / 2.)
        thoucond = thoucond[:,
                            med_ener - ceil(nener / 2):
                            med_ener + floor(nener / 2)]

    # arithmetic mean of samples
    # -> first arithmetic mean for each sample
    # then arithmetic mean across samples
    ari_ari_mean = np.mean(thoucond, axis=None)

    # first the geometric mean inside a sample,
    # then the arithmetic mean across samples
    ari_geo_mean = np.mean(gmean(thoucond, axis=1))

    #
    geo_ari_mean = gmean((np.mean(thoucond, axis=1)))

    geo_geo_mean = gmean(thoucond, axis=None)

    output = (ari_ari_mean, ari_geo_mean,
              geo_ari_mean, geo_geo_mean,
              ari_ari_mean * hilbert_dim,
              ari_geo_mean * hilbert_dim,
              geo_ari_mean * hilbert_dim,
              geo_geo_mean * hilbert_dim,
              nener, hilbert_dim)

    output = tuple(map(np.atleast_1d, output))
    return output


def thoucond_ave(h5file, results_key='Spectrum_differences',
                 disorder_key='dW',
                 target_variance=1. / 3.,
                 epsilon=0.,
                 dW_min=0.,
                 population_variance=True,
                 mode=0,
                 disorder_string='Hamiltonian_random_disorder_partial',
                 sample_averaging=True,
                 nener=-1,
                 *args,
                 **kwargs):
    """
    A routine for performing statistical analysis of the ipr (inverse
    participation ratio) results.

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

    ari_ari: float TO DO: add explanations

    ari_geo: float

    geo_ari: float

    geo_geo: float

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
                           _thoucond_ave, 10,
                           sample_averaging=sample_averaging,
                           nener=nener,
                           *args,
                           **kwargs)
    return results
