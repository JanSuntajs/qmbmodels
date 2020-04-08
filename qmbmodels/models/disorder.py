"""
Provides functions for creating the most commonly
used disorder distributions.

In general, apart from the system size L and
disorder type, each disorder distribution
should have two key parameters:

W: center of the disorder distribution
dW: width of the disorder distribution

For disorder distributions centered at the
origin, the dependence of the variance on
the dW parameter should be the following:

Uniform:

    rho^2 = dW^2 / 3

Binary:

    rho^2 = dW^2

Gaussian:

    rho^2 = dW^2

Cosuniform (cosine of a random number:

    rho^2 = dW^2 / 2


"""
import numpy as np

_available_disorders = ['none', 'uniform', 'binary', 'gaussian',
                        'incomm', 'cosuniform', 'powerlaw']
"""array_like: specifies which types of disorder are currently implemented."""


def get_disorder_dist(L, disorder_type='none', *args, dim=1, **kwargs):
    """
    A function that returns some of the most commonly
    used disorder types used in our calculations. The
    most general usage would be the simulation of
    random potential disorder and/or random exchange
    couplings.

    Parameters:
    -----------
    L: int
            Integer specifying the system size, or, more
            specifically, the number of the random coupling
            constants.

    disorder_type: string, optional
                String specifying from which distribution the
                disordered values should be sampled. Currently
                allowed:
                'uniform', 'binary', 'gaussian', 'incommensurate',
                'cosuniform' (cosine of a uniform variable)
                'none'
                NOTE: incomm type is currently only implemented
                in the 1D case
    dim: int, optional
                Integer specifying the system's dimensionality.
                Defaults to 1 as higher dimensions are only
                applicable in the Anderson noninteracting case
                for now.

    *args: tuple
            Specifying disorder-specific parameters.
            Some common structure should be considered:

            args[0] -> W is usually the center of the
            disorder distribution
            args[1] -> dW is usually the width of the
            disorder distribution
            args[-1] -> seed. The random seed integer
            should always come last.

    Returns:
    --------
    disorder: ndarray
            Array with the random field values.

    Raises:
    -------
    ValueError
        If disorder_type is not recognized.

    """

    if disorder_type not in _available_disorders:
        err_message = ('Disorder type {} not yet ',
                       'implemented. Available types: ',
                       '{}').format(disorder_type, _available_disorders)
        raise ValueError(err_message)

    size = tuple(L for i in range(dim))
    # this try/except block handles the cases
    # in which all the args are not specified
    # because they may not be needed to initialize
    # that particular disorder type
    try:
        W = args[0]
        dW = args[1]
        seed = args[-1]
        np.random.seed(seed=seed)
    except IndexError:
        pass

    if disorder_type == 'none':

        disorder = np.zeros(size, dtype=np.float64)

    if disorder_type == 'uniform':

        disorder = np.random.uniform(
            W - dW, W + dW, size=size)

    if disorder_type == 'binary':

        disorder = np.random.choice([W - dW, W + dW], size=size)

    if disorder_type == 'gaussian':

        disorder = np.random.normal(loc=W, scale=dW, size=size)

    if disorder_type == 'incomm':

        if dim != 1:
            raise ValueError(('incomm disorder type '
                              'only works in 1 dimension!'))

        gldn_ratio = 0.5 * (np.sqrt(5.) - 1.)

        rnd_phase = np.random.uniform(0., 2 * np.pi)

        lattice = np.arange(1, L + 1, 1)
        disorder = dW * \
            np.cos(2 * np.pi * gldn_ratio * lattice + rnd_phase)

    if disorder_type == 'cosuniform':

        # randomly distributed disorder -> site dependent
        # disorder on a lattice
        lattice = np.random.uniform(0., 2 * np.pi, size=size)

        disorder = dW * np.cos(lattice)

    if disorder_type == 'powerlaw':

        disorder = np.random.uniform(0., 1., size=size)

        disorder = dW * disorder ** (1. / W)

    return disorder
