import numpy as np
import h5py

from .disorder import reduce_variance, disorder_analysis


footer_entro = """
Each row is organised as follows:

dW: disorder strength

average_entropy S: average entropy for a given number of states and samples
rescaled entropy S_re: |log(2) - 2**(2*LA - L - 1) / LA - S/LA|; L-> system,
LA-> subsystem

Delta S: standard deviation of S

size L: system size

nener: number of energies obtained using partial diagonalization

target_variance: Variance of the disordered samples
with which to compare the numerical results in the postprocessing
steps if mode equals 1 or 2. Defaults to None as the argument is not
required in the mode=0 case where postprocessing is not performed.

epsilon: condition used to determine whether to select a given disorder
distribution.
If mode=1:

If mode=2:


variance_before: variance of variances of the disorder distributions before
post.

variance_after: variance of variances of the disorder distributions after post.

nsamples: number of all the random samples

nsamples_selected: number of the random disorder samples with an appropriate
variance

nsamples_rejected: nsamples - nsamples_selected

mode: which postprocessing mode was selected

population_variance: integer specifying whether theoretical prediction for
the population variance was used in order calculate the target variance.
1 if that is the case, 0 if not.
"""


def entro_ave(h5file, results_key='Entropy',
              disorder_key='dW',
              target_variance=1. / 3.,
              epsilon=0., population_variance=True,
              mode=0,
              disorder_string='Hamiltonian_random_disorder_partial',
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

    population_variance: boolean
    If True, target variance is the population variance of the
    disorder distribution. In that case, the value of the target_variance
    argument should be a numerical prefactor to the square of the
    disorder parameter's strength: target_variance = dW**2 * target_variance
    If False, target_variance is calculated as the mean of the distribution
    of the sample variances.

    results_key: string, optional
    String specifying which results to load from the .hdf5 file. The argument
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
    in case the key under which the disorder samples are stored in the .hdf5
    files might change in some manner.

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

    dW = nsamples = nener = ave_entro = std_entro \
        = size = entro_rescaled = nsamples_rejected \
        = nsamples_selected \
        = std_before = std_after = None

    try:

        with h5py.File(h5file, 'r') as file:

            key = results_key

            if ((disorder_string in file.keys()) and (key in file.keys())):

                disorder = file[disorder_string][()]
                entropy = file[key][()]
                nsamples = file[key].attrs['nsamples']
                nener = file[key].attrs['nener']
                size = file[key].attrs['L']
                dW = np.float(file[key].attrs[disorder_key])

                if not bool(mode):
                    if population_variance:

                        target_variance *= dW**2
                    else:

                        means_, variances, *rest = disorder_analysis(disorder,
                                                                     size)
                        target_variance = np.mean(variances)
                else:
                    epsilon = None
                    target_variance = 0

                # reduce variance
                (condition, nsamples_dis, nsamples_selected,
                 std_before, std_after) = reduce_variance(disorder, mode,
                                                          size,
                                                          target_variance,
                                                          epsilon)

                check_shapes = (nsamples == nsamples_dis)

                if check_shapes:
                    sub = size / 2.
                    entropy = entropy[condition]
                    ave_entro = np.mean(entropy)
                    entro_rescaled = np.abs(
                        np.log(2) - (2**(2 * sub - size - 1)) / sub -
                        ave_entro / sub)
                    std_entro = np.std(entropy)

                    nsamples_rejected = nsamples - nsamples_selected
                else:

                    print('Shape mismatch! Check the file: {}'.format(h5file))

            else:

                print('Key {} or {} not present in the HDF5 file!'.format(
                    key, disorder_string))

    except IOError:

        print('File {} not present!'.format(h5file))

    return (dW, ave_entro, entro_rescaled, std_entro, size, nener,
            target_variance, epsilon, std_before,
            std_after, nsamples, nsamples_selected, nsamples_rejected,
            mode, bool(population_variance))
