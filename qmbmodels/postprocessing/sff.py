import numpy as np
import h5py

from .disorder import _preparation


footer_sff = """
Formulae for the calculation of the spectral form factor
(SFF) K(tau):

Spectral form factor with the included disconnected part:

K(tau)_uncon = < | \sum_n g(E_n) exp(-1j E_n \tau) | ^2 >

Here, < ... > is the disorder averaging, E_n are the eigenvalues
and g(E_n) is a filter used to minimize the finite-size effects.

To obtain the connected spetral form factor K(tau)_conn, we need
to subtract the disconnected part, which is defined as follows:

uncon(tau) = |< \sum g(E_n) exp(-1j E_n \tau) >|^2

Below we explain the contents of this file.

Each row is organised as follows:

0) dW: disorder strength

1) taulist: values of the (unfolded, hence nonphysical) tau
   parameter, rescaled in such a way that the plateau of
   the sff dependence take place at tau=1.

2) K(tau)_uncon: SFF dependence with the disconnected part included,
   rescaled so that K(tau -> infinity) equals 1. The rescaling
   factor D' is calculated as follows:
   D' = < \sum_n |g(E_n)|^2 >
   Here, g(E_n) is the filter used in the SFF filtering and
   E_n are the eigenvalues. We see that this normalization
   parameter is simply the diagonal part of the K(tau)_uncon
   dependence.

3) K(tau)_conn: SFF dependence without the disconnected part
   and normalized so that the quantity should equal
   K(tau=0) = 0.
   Normalization is as follows:
   (1/D') (K(tau)_uncon- (A/B) * uncon(tau))
   Here, A and B are the values of K(0)_uncon and uncon(0), respectively.
   Hence,
   A = < | \sum_n g(E_n) |^2 >
   B = |< \sum g(E_n) >|^2

4) nener0: number of energies obtained using the diagonalization routine.
   If some parts of the spectrum were discarded during the unfolding
   procedure, the actual number of the energies used in the calculation
   might be smaller.

5) gamma: width of the original spectrum for which we assume a Gaussian
   shape.

6) unfolding_n: degree of the unfolding polynomial used in the
   unfolding procedure.

7) discard_unfolding: number of energies that have been discarded due
   to the unfolding procedure if slope needed to be corrected ->
   unfolding can sometimes introduce unphysical effects near the spectral
   edges with the slope (density of states) becoming negative -> we
   need to cut those edges, thus discarding some values.

8) size L: system size

9) nener: number of energies obtained using partial diagonalization

10) target_variance: Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2.
    If mode=0: nan, this argument is not needed if preprocessing is not
    performed.

11) epsilon: condition used to determine whether to select a given disorder
   distribution.
   If mode=0: nan

12) dW_min: value of the disorder strength parameter for which the epsilon
   was evaluated.
   If mode=0: nan

13) variance_before: variance of variances of the disorder distributions before
    post.

14) variance_after: variance of variances of the disorder distributions after
    post.

15) nsamples: number of all the random samples

16) nsamples_selected: number of the random disorder samples with an
    appropriate variance.
    NOTE: if mode (entry 15) ) equals 0, nsamples equals nsamples.

17) nsamples_rejected: nsamples - nsamples_selected
    NOTE: if mode (entry 15) ) equals 0, this should be equal to 0.

18) mode: which postprocessing mode was selected
    NOTE: 0 indicates no postprocessing!

19) population_variance: integer specifying whether theoretical prediction for
    the population variance was used in order calculate the target variance.
    1 if that is the case, 0 if not.
"""


def _sff(spectrum, condition, size, eff_dims,
         normal_con, normal_uncon,
         gamma0, nener0,
         unfolding_n,
         discarded_unfolding,
         filter_eta,
         *args, **kwargs):

    # taulist rescaled by 2*np.pi so that the plateau
    # is reached at tau = 1 (this holds for the unfolded spectra)
    taulist = spectrum[0] / (2 * np.pi)
    # rescaled sff with the included disconnected
    # part
    sff_disconn = spectrum[1] / eff_dims

    # remove the disconnected part and rescale so that
    # the start is at zero:
    sff_conn = (spectrum[1] - (normal_con / normal_uncon) *
                spectrum[2]) / eff_dims

    gamma = np.ones_like(sff_conn) * gamma0
    nener = np.ones_like(sff_conn) * nener0
    unfolding_n = np.ones_like(sff_conn) * unfolding_n
    discarded_unfolding = np.ones_like(sff_conn) * discarded_unfolding
    filter_eta = np.ones_like(sff_conn) * filter_eta
    return (np.atleast_1d(taulist),
            np.atleast_1d(sff_disconn), np.atleast_1d(sff_conn),
            nener, gamma, unfolding_n, discarded_unfolding, filter_eta)


def get_sff(h5file, results_key='SFF_spectrum',
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

    with h5py.File(h5file, 'r') as file:

        eff_dims = file[results_key].attrs['dims_eff']
        normal_con = file[results_key].attrs['normal_con']
        normal_uncon = file[results_key].attrs['normal_uncon']
        gamma0 = file[results_key].attrs['gamma0']
        nener0 = file[results_key].attrs['nener0']
        discarded_unfolding = file[results_key].attrs['discarded_unfolding']
        unfolding_n = file[results_key].attrs['sff_unfolding_n']
        filter_eta = file[results_key].attrs['eta']

    return _preparation(h5file, results_key, disorder_key,
                        disorder_string,
                        target_variance, population_variance,
                        mode, epsilon, dW_min,
                        _sff, 5, eff_dims=eff_dims,
                        normal_con=normal_con,
                        normal_uncon=normal_uncon,
                        gamma0=gamma0,
                        nener0=nener0,
                        unfolding_n=unfolding_n,
                        discarded_unfolding=discarded_unfolding,
                        filter_eta=filter_eta,
                        *args, **kwargs)
