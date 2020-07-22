import numpy as np
from lmfit.models import LinearModel

from .naive import naive_mi_1d
from .fitting_utils import fit_linear_to_array


def create_samples_1d(
    x, y,
    bin_sizes, fractions, num_samples,
):
    """
    Direct method for the MI estimation proposed by Slonim et al., 2008.

    x, y: Arrays of shape (N,) representing the 1d random variables.
    bin_sizes: bin sizes to take for lim -> inf
    fractions: fractions to use
    num_samples: how many draws to make for each fraction and bin size to
        calculate mean and std from.
    Returns:
        Mutual information estimate
    """
    MI_samples = np.empty((len(bin_sizes), 2, len(fractions), num_samples))
    for i in range(len(bin_sizes)):
        for j in range(len(fractions)):
            for k in range(num_samples):
                inds = np.random.choice(
                    x.shape[0],
                    int(fractions[j]*x.shape[0]),
                    replace=False,
                )
                # print(x[inds].shape)
                MI_samples[i, 0, j, k] = naive_mi_1d(
                    x[inds], y[inds],
                    (bin_sizes[i], bin_sizes[i])
                )[0]
                MI_samples[i, 1, j, k] = naive_mi_1d(
                    x[inds], np.random.permutation(y[inds]),
                    (bin_sizes[i], bin_sizes[i])
                )[0]
    MIs = np.stack([
        MI_samples.mean(axis=-1),
        MI_samples.std(axis=-1),
    ], axis=-1)
    return MIs


def direct_method_1d(
    x, y, bin_sizes, fractions, num_samples,
    shuffle_threshold=0.02, return_slope=False,
):
    MIs = create_samples_1d(x, y, bin_sizes, fractions, num_samples)
    MIs = MIs.reshape((-1, len(fractions), 2))

    params = fit_linear_to_array(MIs[..., 0], 1/fractions, MIs[..., 1])
    params = params.reshape((len(bin_sizes), 2, 4))

    where_less = params[:, 1, 2] < shuffle_threshold

    model = LinearModel()
    result = model.fit(
        data=params[where_less, 0, 2],
        x=1/bin_sizes[where_less]**2,
        weights=params[where_less, 0, 3]**2,
    )
    mi = result.params["intercept"].value
    mi_err = result.params["intercept"].stderr
    if return_slope:
        return mi, mi_err, result.params["slope"].value
    else:
        return mi, mi_err
