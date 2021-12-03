#!/usr/bin/env python

"""
    This module contains helper routines for calculations
    of spectral functions from the Hamiltonian matrix elements
    as well as susceptibilities.

    """
import numpy as np
import numba as nb


@nb.njit()
def _running_mean(arr, n_window):
    """
    An internal routine for calculating the
    running mean of some dataset. Taken from
    here:

    https://stackoverflow.com/questions/14313510/
    how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """

    ret = np.cumsum(arr)
    ret[n_window:] = ret[n_window] - ret[:-n_window]

    return ret[n_window - 1:] / (n_window * 1.)


def _smoothing(diffs, matelts, n_window):
    """
    A smoothing function for the matelts which
    is already sorted in accordance to the entries
    of the diffs array. First, the last
    matelts.size % n_window values are discarded
    to allow for reshaping of the array. Then, the
    input arrays are reshaped as
    reshaped = input.reshape(-1, n_window); then, mean
    over axis=1 of the reshaped arrays is taken to
    smoothen the results.

    Parameters:

    diffs: 1d ndarray, dtype=np.float64
    A sorted array of positive real numbers
    specifying the absolute values of the
    energy differences of the corresponding
    matrix elements (matelts).

    matelts: 1d ndarray, dtype=np.float64
    An array matching the shape of the diffs
    array (each entry in the matelts array
    corresponds to an energy difference in the
    diffs array).

    n_window: int
    To smoothen the results, we bin the values
    in the input arrays together, putting n_window
    values in each bin.

    Returns:

    diffs: 1D ndarray, dtype=np.float64
    A smoothened diffs array.

    matelts: 1D ndarray, dtype=np.float64
    A smoothened matelts array.
    """

    n_elts = diffs.size
    n_remain = n_elts % n_window

    if n_remain != 0:
        diffs = diffs[: -n_remain]
        matelts = matelts[: -n_remain]
    diffs = np.mean(diffs.reshape(-1, n_window), axis=1)
    matelts = np.mean(matelts.reshape(-1, n_window), axis=1)

    return diffs, matelts


def calc_susceptibility(eigvals, matelts, mu):
    """
    Evaluate the susceptibility of the eigenstates based
    on the definitions in the following two papers:

    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.041017
    (Adiabatic eigenstate deformations as a sensitive probe for
    quantum chaos, Pandey, Clayes, Campbell, Polkovnikov, Sels, 
    PRX 10, 04017, 2020)
    and 
    https://arxiv.org/pdf/2105.09348.pdf
    (Thermalization through linked conducting clusters in spin
    chains with dilute effects, Sels, Polkovnikov).

    The former paper introduces the concept of an adiabatic gauge
    potential (AGP), and the
    notion of filtering which might be required to mitigate the effects
    of near-degeneracies leading to nearly-diverging values of the AGP.
    The latter paper introduces the concept of the fidelity
    susceptibility which is simply the contribution of an individual
    eigenstate to the AGP sum. Npte: we report values of the susceptibilities
    rescaled by the Hilbert space dimension.

    Parameters:

    eigvals: 1d ndarray, dtype=np.float64
    Array of the Hamiltonian eigenvalues.

    matelts: 2d ndarray, dtype = np.float64

    mu: float
    The filtering term to smoothen out the near-divergences (to
    regularize the sum.)

    Returns:

    susc: 1d ndarray, float
    An array of the calculated susceptibilities for zero filtering.

    log_susc: 1d ndarray, float
    Logarithms of the susceptibilites in the susc array.

    susc_filt: 1d ndarray, float
    Susceptibility calculated for a selected filter.

    log_susc_filt: 1d ndarray, float
    Logarithm of susceptibilities calculated for a selected filter. 

    """

    # energy difference array
    diffs = np.abs(eigvals[:, np.newaxis] - eigvals)**2

    # mu = np.mean(np.diff(eigvals)) * size
    np.fill_diagonal(diffs, np.nan)

    # leave the susceptibilies as an array
    susc = np.nansum(
        np.abs(matelts)**2/diffs, axis=1)
    log_susc = np.log(susc)

    susc_filt = np.nansum(np.abs(matelts**2)*diffs / (mu**2 + diffs)**2, axis=1)

    log_susc_filt = np.log(susc_filt)

    return susc, log_susc, susc_filt, log_susc_filt


def eval_spectral_fun(eigvals, matelts, target_ene, eps,
                      full_spectrum=True):
    """
    A function for the evaluation of the spectral function as well
    as other related quantities. NOTE: no smoothing is performed
    inside this function as it can be performed afterwards on the
    obtained results.

    Parameters:

    eigvals: 1D ndarray, dtype=np.float64
    An array of eigenvalues e_i corresponding to the
    matrix elements M_{i,j}.

    matelts: 2D ndarray, dtype=np.float64
    An array of matrix elements M_{i, j} of shape
    (eigvals.size, eigvals.size)

    target_ene: np.float64
    Only relevant if full_spectrum==False in which case
    only matrix elements from a chosen energy window
    are selected.
    Target energy around which the matrix elements
    corresponding to selected eigenpairs are evaluated.
    For a matrix element M_{i,j} the mean energy of the
    pair of eigenvalues e_ij = 0.5 * (e_i + e_j) should be
    close enough to target_ene for the matrix element
    to be considered.

    eps: np.float64
    Only relevant if full_spectrum==False in which case
    only matrix elements from a chosen energy window are
    selected. 
    The parameter eps determines the width of the said energy
    window according to the rule: e_width = eps * (e_{-1} - e_0).
    To select a given matrix element, one must have
    ((e_ij < target_ene + e_width) & (e_ij > target_ene - e_width)).
    See also the entry for target_ene.


    full_spectrum: boolean, optional
    Whether to perform the analysis of the matrix elements within
    a small energy window or whether to consider all of them. Defaults
    to True (hence to the latter possibility).

    Returns:

    diagonals

    diffs

    matelts

    matelts_integ

    n_vals

    target_ene

    eps_

    bandwidth

    """
    #
    bandwidth = eigvals[-1] - eigvals[0]
    eps_ = eps * bandwidth

    # energy difference array
    diffs = (eigvals[:, np.newaxis] - eigvals)

    # first, extract the diagonal matrix elt
    diagonals = np.diagonal(matelts).copy()

    # if no energy window is applied, pick
    # everything except the diagonals
    if full_spectrum:

        boolar = np.ones_like(matelts, dtype=np.bool_)
        np.fill_diagonal(boolar, False)

    # in the opposite case, pick the states from
    # the energy window not on the diagonal
    else:

        boolar = 0.5 * (eigvals[:, np.newaxis] + eigvals)
        boolar = ((boolar < target_ene + eps_) &
                  (boolar > target_ene - eps_))
        np.fill_diagonal(boolar, False)

    diffs = np.abs(diffs[boolar].flatten())
    n_vals = diffs.size
    sort_args = np.argsort(diffs)
    diffs = diffs[sort_args]
    matelts = matelts[boolar].flatten()
    matelts = np.abs(matelts[sort_args])**2
    # for the full spectrum, also calculate the properly
    # normalized integrated spectral function
    if full_spectrum:
        matelts_integ = np.cumsum(matelts) / eigvals.size
        # diffs_, matelts_integ = _smoothing(
        #     diffs, matelts_integ, window_smoothing)
    else:
        matelts_integ = np.array([])
    #diffs, matelts = _smoothing(diffs, matelts, window_smoothing)
    return (diagonals, diffs, matelts, matelts_integ, n_vals,
            target_ene, eps_, bandwidth)


@nb.njit("Tuple((float64[:], float64[:], float64[:], float64[:]))(float64[:], float64[:, :], int64)", fastmath=True)
def eval_matelt_variances(eigvals, matelts, n_window):
    """
    A function for evaluating the variances
    of the diagonal and offdiagonal matrix
    elements. The function basically slides
    a window of shape=(n_window, n_window)
    along the diagonal of the matelts array.
    On each step, the microcanonical average
    energy of the window is calcualted, as well
    as the variances of diagonal and offdiagonal
    matrix elements. Finally the ratio of the last
    two quantities is calculated as well.

    Parameters:

    eigvals: 1d ndarray, float64
    Eigenvalues corresponding to the matrix elements.

    matelts: 2d ndarray, float64
    Matrix elements array to be analysed.

    n_windows: int
    Width of the window in which the properties
    are to be investigated

    Returns:

    mean_ener, vars_diag, vars_offdiag, ratios: 1d ndarrays, float64
    Arrays of microcanonical energies, diagonal matrix elements variances,
    offdiagonal matrix elements variances and the ratios of the last two
    quantities, respectively. Values are given for each consecutive window
    position.

    """

    mean_ener = _running_mean(eigvals, n_window)

    vars_diag = np.zeros_like(mean_ener)
    vars_offdiag = np.zeros_like(mean_ener)
    ratios = np.zeros_like(mean_ener)

    mask = np.array([[True for j in range(n_window)] for i in range(n_window)])
    np.fill_diagonal(mask, False)
    mask = mask.flatten()
    for i in range(matelts.shape[0] + 1 - n_window):

        mat = matelts[i:i+n_window, i:i+n_window]
        diag = np.diag(mat).copy()
        var_diag = np.std(diag)**2
        offdiag = mat.flatten()[mask]
        var_offdiag = np.std(offdiag)**2

        ratio = var_diag * 1. / var_offdiag

        vars_diag[i] = var_diag
        vars_offdiag[i] = var_offdiag
        ratios[i] = ratio

    return mean_ener, vars_diag, vars_offdiag, ratios
