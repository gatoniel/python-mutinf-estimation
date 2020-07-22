import numpy as np
from sklearn.neighbors import KDTree
from scipy.special import digamma


def estimator1(x, y, k):
    """
    Estimator 1 of Estimating mutual information, A. Kraskov et al., Physical
    Review E 69, 2004.
    x, y: Arrays of shape (N, Dx) and (N, Dy), where N is the number of samples
    and Dx, Dy are the dimensions of the random variables X and Y.
    k: k for the k-th nearest neighbors

    returns the estimate for the mutual information
    """
    z = np.concatenate([x, y], axis=-1)
    # we use chebyshev/maximum metric, since it is the
    # easiest to fulfill eq. 6 of the paper in all dimensions
    tree = KDTree(z, metric="chebyshev")
    # We need to add 1, since query returns the identity
    dist, ind = tree.query(z, k + 1)
    dist = dist[:, -1]
    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")
    # query radius with count_only=True returns one count too much
    # for one of the subspaces, since it is not using strictly less
    indx, distx = tree_x.query_radius(
        x, dist, return_distance=True, count_only=False
    )
    indy, disty = tree_y.query_radius(
        y, dist, return_distance=True, count_only=False
    )

    distxy = [distx, disty]
    counts = np.empty((x.shape[0], 2))
    for i in range(x.shape[0]):
        for j in range(2):
            tmp_dist = distxy[j][i]
            less = tmp_dist < dist[i]
            counts[i, j] = less.sum()

    # we do not need to add 1, since query_radius allready counted
    # the point itself. So counts[:, i] is allready one too much.
    digamma_x_mean = digamma(counts[:, 0]).mean()
    digamma_y_mean = digamma(counts[:, 1]).mean()
    return digamma(k) + digamma(x.shape[0]) - digamma_x_mean - digamma_y_mean


def estimator2(x, y, k):
    """
    Estimator 2 of Estimating mutual information, A. Kraskov et al., Physical
    Review E 69, 2004.
    x, y: Arrays of shape (N, Dx) and (N, Dy), where N is the number of samples
    and Dx, Dy are the dimensions of the random variables X and Y.
    k: k for the k-th nearest neighbors

    returns the estimate for the mutual information
    """
    z_list = [x, y]
    z = np.concatenate(z_list, axis=-1)
    # we use chebyshev/maximum metric, since it is the
    # easiest to fulfill eq. 6 of the paper in all dimensions
    tree = KDTree(z, metric="chebyshev")
    # We need to add 1, since query returns the identity
    ind = tree.query(z, k + 1, return_distance=False)
    ind = ind[:, -1]
    # calculate the distances in the subspaces X and Y
    distx, disty = [np.max(
        np.abs(a - a[ind, :]),
        axis=-1
    ) for a in [x, y]]

    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")

    countsx = tree_x.query_radius(x, distx, count_only=True)
    countsy = tree_y.query_radius(y, disty, count_only=True)

    # we need to substract 1, since query_radius allready counted
    # the point itself.
    digamma_x_mean = digamma(countsx - 1).mean()
    digamma_y_mean = digamma(countsy - 1).mean()
    return (
        digamma(k) + digamma(x.shape[0]) - 1/k
        - digamma_x_mean - digamma_y_mean
    )
