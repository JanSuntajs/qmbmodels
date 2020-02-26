"""
This module contains routines for analysing the statistical properties
of the disorder ansambles used in our calculations.

Functions disorder_analysis(...) and get_min_variance(...)
are intented for the analysis of the actual disorder samples.
The reduce variance routine is a preprocessing function which
allows one an attempt at reducing the variance of the
observables under consideration.

"""

import numpy as np
import h5py

from .utils import extract_single_model, _extract_disorder


def disorder_analysis(samples, system_size):
    """
        A function used to obtain some statistical
        observables of the disordered spectra used
        to prepare our model Hamiltonians.

        Parameters:
        -----------

        samples: ndarray
        A 2D ndarray of the shape (nsamples, system_size)
        with different random disorder realizations

        system_size: int
        system size which should match the samples.shape[1]
        value.

        Returns:
        --------

        means: ndarray
        1D ndarray of means of individual disorder realizations

        variances: ndarray
        1D ndarray of variances of individual disorder realizations
        where the unbiased estimator of the population variance
        is returned.

        var_variances: float
        variance of variances of the individual sample variances

        rescale factor: float
        1. / (system_size - 1) -> the rescaling factor needed in some
        calculations where the unbiased form of the population
        variance is used.


    """

    means = np.mean(samples, axis=1)
    # obtain the variances of individual disorder
    # realizations
    variances = np.var(samples, axis=1, ddof=1)
    # now get the variance of the variances
    std_variances = np.std(variances)

    rescale_factor = np.sqrt(system_size - 1)

    return means, variances, std_variances, rescale_factor


def get_min_variance(topdir, descriptor, syspar, modpar, disorder_key,
                     disorder_string='Hamiltonian_random_disorder_partial'):
    """
        In some calculations, we wish to perform the postprocessing of the
        gathered data in order to reduce the fluctuations of the analysed
        quantities which are a consequence of the finite-size nature of the
        systems under study. This is particulary important in the studies of
        the fluctuations of some quantum/thermodynamic observables, such as
        the fluctuations of the entanglement entropy. In order to ensure an
        'apples to apples' comparison, we wish to make sure the disorder
        distributions of the examined spectra are of the approximately
        equal width (standard deviation/variance). This routine is mostly
        intented for the extraction of some characteristic width of a chosen
        disorder distribution (e.g. the width of the disorder distribution
        for some system size and disorder strength parameter) which is then
        used
        as a benchmark/comparison for other studied system sizes and disorder
        strengths.

        Parameters:
        -----------

        topdir, descriptor, syspar, modpar -> strings, which, in the order
        given here, give the location of the .hdf5 results file:
        /<topdir>/<descriptor>/<syspar>/<modpar>/
        disorder_key: string
        key specifying the substring which stands next to the value of
        the disorder parameter
        disorder_string: string, optional
        which key denotes the disorder in the .hdf5 file

        Returns:

        dW: float
        value of the disorder strength parameter
            other return values: see above for the return of the
            disorder_analysis function

    """

    h5file = extract_single_model(topdir, descriptor, syspar, modpar)

    # get the rest of the string and the value of the disorder parameter
    rest, dW = _extract_disorder(modpar, disorder_key)
    with h5py.File(h5file, 'r') as file:

        disorder = file[disorder_string][()]

        size = file[disorder_string].attrs['L']

        means, variances, std_variances, rescale_factor = disorder_analysis(
            disorder, size)

    return dW, means, variances, std_variances, rescale_factor


def reduce_variance(disorder_samples, mode, size, target_variance, epsilon):
    """
    A procedure for reducing the standard deviation of
    variances of different disorder realizations within
    an ensemble of disordered samples. We implement three
    ways (modes) of variance reduction:

    mode 0:
    No postprocessing, everything is selected.

    mode 1:
    Having an ensemble of <disorder_samples>, which is
    nothing but a 2-dimensional array of different disorder
    realizations for a given system size and disorder strength
    parameter value, we calculate the unbiased estimator of the
    sample variance for each individual sample. Next, we compare
    each sample's variance with
    some target variance (usually thepopulation variance of a given
    disorder distribution) and reject the samples where the difference
    between the sample and target variance exceeds some chosen
    value of epsilon:

        | var(sample) - target_variance | <= epsilon

    The results for various quantum mechanical observables, such as
    the <r> statistic or the entanglement entropy  calculations, are
    then calculated for the spectra with an appropriate disorder
    realization.

    mode 2:
    In the second approach, we again start with calculating the unbiased
    estimators of individual sample variances and proceed by comparing
    them to some selected target variance (usually that would be the
    population variance, as described above). In this case, however, we
    monitor the standard deviation of the sample variances which we wish
    to minimize by succesively rejecting samples with the largest
    difference between the sample and target variance. We stop rejecting
    samples once the standard deviation of the variances of the 'postprocessed'
    disorder realizations begins to match some target standard deviation.

    Parameters:

    disorder_samples: ndarray
    2D ndarray of the shape (nsamples, size) containing the disorder samples
    for different disorder realizations

    mode: int
    Which way of variance reduction to choose, currently modes 0, 1, 2 and 3
    are supported.

    size: int
    System size.

    target_variance: float
    Target variance as described above which is used for comparison/selection
    purpose.

    epsilon: float
    The selection criterion.

    Returns:

    condition: ndarray
    An array of boolean values (for mode 1) or integer indices (for mode 2).
    The condition array can be used to select the values of the observables
    (such as energy spectra, entanglement entropy calculations etc)
    corresponding to the appropriate disorder realizations when performing
    subsequent data analysis. Example:

    suppose we have an array of entanglement entropy results (named
    entropy_arr in the following) of the shape
    (# number of samples, # number of calculated energies)
    where entry
    (m, n)
    in the array corresponds to the entanglement entropy of a n-th
    eigenstate in the m-th disorder realization. By slicing the array
    using
    entropy_arr[condition] we would only select the rows corresponding to the
    disorder realizations with a sufficiently small standard deviation of the
    disorder distribution variances.

    nsamples, nsamples_selected: int
    The number of samples before and after postprocessing, respectively

    std_before, std_after: float
    The standard deviation of the distribution of variances before and
    after postprocessing, respectively.
    """

    def _get_vars(disorder_samples):
        # calculate the unbiased estimators of the sample variances first
        # ddof=1 designates the reduction of the degrees of freedom by 1 ->
        # thus yielding an unbiased estimator for the sample variance
        sample_vars = np.var(disorder_samples, axis=1, ddof=1)
        # subtract the target variance
        sample_vars_sub = sample_vars - target_variance
        # standard_deviation of the variances before postprocessing
        std_before = np.std(sample_vars)

        return sample_vars, sample_vars_sub, std_before

    sample_vars, sample_vars_sub, std_before = _get_vars(disorder_samples)

    std_after = std_before
    # implement the zeroth mode (nothing happens)
    nsamples = disorder_samples.shape[0]
    condition = np.arange(nsamples)

    if mode == 0:

        nsamples_selected = nsamples

    # implement the first mode

    elif mode == 1:

        condition = np.abs(sample_vars_sub) * np.sqrt(size - 1) < epsilon
        # how many samples were selected
        nsamples_selected = np.sum(condition)
        # get std_after
        *blank, std_after = _get_vars(disorder_samples[condition])

    elif mode == 2:

        indices = []

        # indices of the largest deviations
        argsortlist = np.argsort(np.abs(sample_vars_sub))[::-1]
        i = 0
        while std_after * np.sqrt(size - 1) >= epsilon:

            max_arg = argsortlist[i]
            # find the maximum deviation from the target
            indices.append(max_arg)

            # remove the largest sample from the array
            disorder_samples_ = np.delete(disorder_samples, indices, 0)

            sample_vars, sample_vars_sub, std_after = _get_vars(
                disorder_samples_)

            i += 1

        condition = np.delete(condition, indices)
        nsamples_selected = len(condition)

    else:

        print(('Only modes 0, 1 and 2 are currently available!'
               'mode {} not yet implemented!').format(mode))

    return condition, nsamples, nsamples_selected, std_before, std_after


def _preparation(h5file, results_key, disorder_key,
                 disorder_string, target_variance,
                 population_variance, mode,
                 epsilon, dW_min, analysis_fun,
                 analysis_fun_shape,
                 sample_averaging=True,
                 *args,
                 **kwargs):
    """
    A function used for preparation and pre-processing
    of the numerical results used in our subsequent
    analysis. In actual usage, this function is meant
    to be wrapped by some other (wrapper) function.

    Parameters:
    -----------

    Parameters:
    -----------

    h5file: string
    Filename of the .hdf5 file containing the numerical data to be
    analysed.

    results_key: string, optional
    String specifying which results to load from the .hdf5 file.
    The argument
    is provided for compatibility of future versions of this code package
    in case the key under which the entropy results are stored in the .hdf5
    files might change in some manner.

    disorder_key: string, optional
    String specifying which parameter descriptor describes disorder.
    target_variance: {float, None}
    Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2. Defaults to None as the argument is not
    required in the mode=0 case where postprocessing is not performed.
    See also the explanation for the population_variance parameter.

    disorder_string: string, optional
    String specifying the key under which the disordered spectra are
    stored in thethe .hdf5 file. The argument
    is provided for compatibility of future versions of this code package
    in case the key under which the disorder samples are stored in the
    .hdf5 files might change in some manner.

    target_variance: {float, None}
    Variance of the disordered samples
    with which to compare the numerical results in the postprocessing
    steps if mode equals 1 or 2. Defaults to None as the argument is not
    required in the mode=0 case where postprocessing is not performed.
    See also the explanation for the population_variance parameter.

    population_variance: boolean
    If True, target variance is the population variance of the
    disorder distribution. In that case, the value of the target_variance
    argument should be a numerical prefactor to the square of the
    disorder parameter's strength: target_variance =
    dW**2 * target_variance
    If False, target_variance is calculated as the mean of the distribution
    of the sample variances.

    mode: int, optional
    Which of the postprocessing modes to use:
            0: no postprocessing
            1: first variance reduction scheme
            2: second variance reduction scheme
    See the documentation of the disorder.reduce_variance(...) function
    for further details.

    epsilon: {float, None}
    Stopping criterion for the postprocessing variance reduction routines
    if mode=1 or mode=2. In the absence of postprocessing, epsilon is set
    to None.

    dW_min: {float, None}
    the disorder strength parameter value for which the epsilon
    was evaluated. Set to None in the absence of postprocessing.

    analysis_fun: function
    A function from the available_routines._routines_dict which
    is used to process the numerical results. The function should
    have the following interface:
    analysis_fun(result, condition, size, **kwargs)
    Where:
        result: 2D ndarray
        condition: array-like, used to slice the result array
        size: int

    analysis_fun_shape: int
    Length of the analysis_fun output tuple -> the number
    of objects returned by the analysis_fun function.

    sample_averaging: boolean, optional
    Defaults to True, hence the return of the function is
    a tuple of scalar values -> spectral observables are
    calculated for each sample independently and the
    average (or some other moment) of the sample observables
    is then taken.
    If False, no sample averaging is performed and the return
    is a tuple of array-like objects. This is particulary
    suited for analysis functions in which we are interested
    in the dependence of results on particular disorder
    samples/realizations.
    Examples of such an analysis would be:
        - storing spectral observables for each disorder
          realization independently without performing
          the sample averaging
        - Calculating spectral observables as a function
          of the disorder samples included
        - etc.

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

    Returns:
    --------

    dW: float
    Value of the disorder strength parameter

    *results: the output of the analysis_fun function.

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

    # initialize the known output to None
    dW = nsamples = nener = size \
        = nsamples_selected \
        = nsamples_rejected = std_before \
        = std_after = None

    # initalize the output of the analysis_fun
    results = [None for i in range(analysis_fun_shape)]
    nrows = 1
    try:

        with h5py.File(h5file, 'r') as file:

            key = results_key

            if (((disorder_string in file.keys()) or
                 (mode == 0)
                 ) and (key in file.keys())):

                result = file[key][()]
                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                size = file[key].attrs['L']
                dW = np.float(file[key].attrs[disorder_key])

                try:
                    disorder = file[disorder_string][()]
                except KeyError:
                    print(f'Key {disorder_string} does not exist!')
                    disorder = np.zeros((nsamples, size), dtype=np.float)

                if bool(mode):

                    if population_variance:

                        target_variance *= dW**2

                    else:

                        means, variances, *rest = disorder_analysis(disorder,
                                                                    size)

                        target_variance = np.mean(variances)

                else:
                    epsilon = None
                    dW_min = None
                    target_variance = None

                (condition, nsamples_dis, nsamples_selected,
                 std_before, std_after) = reduce_variance(disorder, mode,
                                                          size,
                                                          target_variance,
                                                          epsilon)

                nsamples_rejected = nsamples - nsamples_selected
                check_shapes = (nsamples == nsamples_dis)

                if ((mode == 0) and (not check_shapes)):
                    condition = np.arange(nsamples)
                    std_before = np.NaN
                    std_after = np.NaN
                    check_shapes = True
                if check_shapes:

                    results = analysis_fun(result, condition, size,
                                           sample_averaging=sample_averaging,
                                           *args,
                                           **kwargs)
                    nrows = len(results[0])

                else:

                    print('Shape mismatch! Check the file {}'.format(h5file))

            else:

                print('Key {} or {} not present in the HDF5 file!'.format(
                    key, disorder_string))

    except IOError:

        print('File {} not present!'.format(h5file))

    nres = len(results)
    ncols = nres + 13
    output = np.zeros((nrows, ncols))

    vals = (dW, *results, size, nener, target_variance, epsilon, dW_min,
            std_before, std_after, nsamples, nsamples_selected,
            nsamples_rejected, mode, bool(population_variance))

    for i in range(ncols):

        output[:, i] = vals[i]

    return output


def _preparation_analysis(h5file, results_key, disorder_key,
                          disorder_string, target_variance,
                          population_variance, analysis_fun,
                          *args,
                          **kwargs):
    """
    A function used for numerical analysis
    of the calculation results w.r.t. to
    the number of different disorder
    samples included.

    This private function is meant to be used
    wrapped by some external functions. The results
    are calculated iteratively by first sorting
    the samples according to the deviation of their
    disorder variance w.r.t. the theoretical variance.
    On each step, samples with the largest discrepancy
    are discarded and the observables of interest
    are recalculated.

    Parameters:
    ----------
    """

    output = []

    try:

        with h5py.File(h5file, 'r') as file:

            key = results_key

            if ((disorder_string in file.keys()) and (key in file.keys())):

                disorder = file[disorder_string][()]
                result = file[key][()]
                # this needs to be present in case
                # number of the result files and
                # the number of disorder samples
                # differ, due to, say, some error
                # encountered during the saving process.
                # In case the values differ, stop
                # the analysis
                nsamples = file[key].attrs['nsamples']
                nsamples_dis = file[disorder_string].attrs['nsamples']
                nener = file[key].attrs['nener']
                size = file[key].attrs['L']
                dW = np.float(file[key].attrs[disorder_key])

                if population_variance:

                    target_variance *= dW**2
                else:
                    means_, variances, *rest = disorder_analysis(disorder,
                                                                 size)

                    target_variance = np.mean(variances)

                # the analysis part:
                condition = np.arange(nsamples)
                indices = []
                output = []

                (means, variances, std_variances,
                    rescale_factor) = disorder_analysis(disorder, size)

                variances_sub = np.abs(variances - target_variance)
                argsortlist = np.argsort(variances_sub)[::-1]

                check_shapes = (nsamples == nsamples_dis)

                if check_shapes:

                    for i in range(nsamples):

                        indices.append(argsortlist[i])
                        condition_ = np.delete(condition, indices)

                        disorder_ = disorder[condition_]

                        (*rest, std_variances_,
                            rescale_factor_) = disorder_analysis(
                            disorder_, size)

                        result_ = result[condition_]
                        output_ = analysis_fun(result_, None, size,
                                               sample_averaging=True,
                                               *args,
                                               **kwargs)

                        output_ = np.concatenate(output_)
                        nsamples_selected = nsamples - (i + 1)

                        output.append([i + 1, nsamples_selected,
                                       *output_, target_variance,
                                       bool(population_variance), dW])
                else:

                    print('Shape mismatch! Check the file: {}'.format(h5file))

            else:

                print('Key {} or {} not present in the HDF5 file!'.format(
                    key, disorder_string))

    except IOError:

        print('File {} not present!'.format(h5file))

    return np.array(output)
