"""
This model provides functions for creating
the most commonly used disorder distributions.
In general, apart from the system size L and
disorder type, each disorder distribution
should have two key parameters:

W: center of the disorder distribution
dW: width of the disorder distribution

For disorder distributions centered at the
origin, the dependence of the variance on
the dW parameter should be the following:

uniform:

    rho^2 = dW^2 / 3

binary:

    rho^2 = dW^2

Gaussian:

    rho^2 = dW^2

Quasirandom:

    rho^2 = dW^2 / 2


"""
import numpy as np

_available_disorders = ['none', 'uniform', 'binary', 'gaussian', 'quasirandom']


def get_disorder_dist(L, disorder_type='none', *params):
    """
    A function that returns some of the most commonly
    used disorder types used in our calculations. The
    most general usage would be the simulation of
    random potential disorder and/or random exchange
    couplings.

    parameters:

    disorder_type: string, optional
                String specifying from which distribution the
                disordered values should be sampled. Currently
                allowed:
                'uniform', 'binary', 'gaussian', 'quasirandom',
                'none'
                Defaults to 'none'
        L: int
                Integer specifying the system size, or, more
                specifically, the number of the random coupling
                constants.

        params: tuple
                Specifying disorder-specific parameters.
                Some common structure should be considered:

                params[0] -> W is usually the center of the
                disorder distribution
                params[1] -> dW is usually the width of the
                disorder distribution
                params[-1] -> seed. The random seed integer
                should always come last.

        returns:

        disorder: ndarray
                Array with the random field values.

    """

    if disorder_type not in _available_disorders:
        err_message = ('Disorder type {} not yet ',
                       'implemented. Available types: ',
                       '{}').format(disorder_type, _available_disorders)
        raise ValueError(err_message)

    try:
        W = params[0]
        dW = params[1]
        seed = params[-1]
    except IndexError:
        pass

    np.random.seed(seed=seed)
    if disorder_type == 'none':

        disorder = np.zeros(L, dtype=np.float64)

    if disorder_type == 'uniform':

        disorder = np.random.uniform(
            W - dW, W + dW, size=L)

    if disorder_type == 'binary':

        disorder = np.random.choice([W - dW, W + dW], size=L)

    if disorder_type == 'gaussian':

        disorder = np.random.normal(loc=W, scale=dW, size=L)

    if disorder_type == 'quasirandom':

        gldn_ratio = 0.5 * (np.sqrt(5.) - 1.)

        rnd_phase = np.random.uniform(0., 2 * np.pi)

        lattice = np.arange(1, L + 1, 1)
        disorder = W * \
            np.cos(2 * np.pi * gldn_ratio * lattice + rnd_phase)

    return disorder
