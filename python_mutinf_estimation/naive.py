import numpy as np
from scipy.stats import entropy


def naive_mi_1d(x, y, bin_sizes, weights=None, base=2):
    """
    Calculates the naive Mutual information between
    two 1-dimensional random variables x and y by
    binning them into bin_sizes bins.

    x, y: array_like, shape (N,)
        Arrays containing the x/y coordinates of the points.
    bins_sizes: tuple of length two. Giving the bin sizes for
    x/y values.

    Returns mutual_information, entropy of X, Y, and (X,Y)
    """
    H, _, _ = np.histogram2d(x, y, bin_sizes, density=True, weights=weights)
    H_XY = entropy(H.flatten(), base=base)
    H_X = entropy(H.sum(axis=1), base=base)
    H_Y = entropy(H.sum(axis=0), base=base)
    MI = H_X + H_Y - H_XY
    return MI, H_X, H_Y, H_XY


def naive_mi_1d_dd(x, y, bin_sizes, weights=None, base=2):
    """
    Calculates the naive Mutual information between
    a 1-dimensional random variable x and a D-dimensional random variable y by
    binning them into bin_sizes bins.

    x, y: array_like, shape (N,) and (N, D)
        Arrays containing the x/y coordinates of the points.
    bins_sizes: tuple of length two. Giving the bin sizes for
    x/y values.

    Returns mutual_information, entropy of X, Y, and (X,Y)
    """
    sample = np.empty((x.shape[0], y.shape[1] + 1))
    sample[:, 0] = x
    sample[:, 1:] = y
    bin_sizes = [bin_sizes[0], ] + [bin_sizes[1], ] * y.shape[1]
    H, _, = np.histogramdd(sample, bin_sizes, density=True, weights=weights)
    H_XY = entropy(H.flatten(), base=base)
    H_X = entropy(H.sum(axis=tuple(range(1, y.shape[1] + 1))), base=base)
    H_Y = entropy(H.sum(axis=0).flatten(), base=base)
    MI = H_X + H_Y - H_XY
    return MI, H_X, H_Y, H_XY
