import numpy as np
from scipy.stats import entropy


def naive_mi_1d(x, y, bin_sizes, base=2):
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
    H, _, _ = np.histogram2d(x, y, bin_sizes, density=True)
    H_XY = entropy(H.flatten(), base=base)
    H_X = entropy(H.sum(axis=1), base=base)
    H_Y = entropy(H.sum(axis=0), base=base)
    MI = H_X + H_Y - H_XY
    return MI, H_X, H_Y, H_XY
