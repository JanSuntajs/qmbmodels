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
                 **kwargs):

    # initialize the known output to None
    dW = nsamples = nener = size \
        = nsamples_selected \
        = nsamples_rejected = std_before \
        = std_after = None

    # initalize the output of the analysis_fun
    results = [None for i in range(analysis_fun_shape)]
    try:

        with h5py.File(h5file, 'r') as file:

            key = results_key

            if ((disorder_string in file.keys()) and (key in file.keys())):

                disorder = file[disorder_string][()]
                result = file[key][()]
                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                size = file[key].attrs['L']
                dW = np.float(file[key].attrs[disorder_key])

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
                    target_variance = 0

                (condition, nsamples_dis, nsamples_selected,
                 std_before, std_after) = reduce_variance(disorder, mode,
                                                          size,
                                                          target_variance,
                                                          epsilon)

                nsamples_rejected = nsamples - nsamples_selected
                check_shapes = (nsamples == nsamples_dis)

                if check_shapes:

                    results = analysis_fun(result, condition, size)

                else:

                    print('Shape mismatch! Check the file {}'.format(h5file))

            else:

                print('Key {} or {} not present in the HDF5 file!'.format(
                    key, disorder_string))

    except IOError:

        print('File {} not present!'.format(h5file))

    return (dW, *results, size, nener, target_variance, epsilon, dW_min,
            std_before, std_after, nsamples, nsamples_selected,
            nsamples_rejected, mode, bool(population_variance))
