import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

from .naive import naive_mi_1d, naive_mi_1d_dd
from .fitting_utils import fit_linear_to_array


def create_samples_1d(
    x, y,
    bin_sizes, fractions, num_samples,
    weights=None,
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
                weights_tmp = None if weights is None else weights[inds]
                MI_samples[i, 0, j, k] = naive_mi_1d(
                    x[inds], y[inds],
                    (bin_sizes[i], bin_sizes[i]),
                    weights=weights_tmp
                )[0]
                MI_samples[i, 1, j, k] = naive_mi_1d(
                    x[inds], np.random.permutation(y[inds]),
                    (bin_sizes[i], bin_sizes[i]),
                    weights=weights_tmp
                )[0]
    MIs = np.stack([
        MI_samples.mean(axis=-1),
        MI_samples.std(axis=-1),
    ], axis=-1)
    return MIs


def create_samples_1d_dd(
    x, y,
    bin_sizes, fractions, num_samples,
    weights=None,
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
                weights_tmp = None if weights is None else weights[inds]
                MI_samples[i, 0, j, k] = naive_mi_1d_dd(
                    x[inds], y[inds, :],
                    (bin_sizes[i], bin_sizes[i]),
                    weights=weights_tmp
                )[0]
                MI_samples[i, 1, j, k] = naive_mi_1d_dd(
                    x[inds], np.random.permutation(y[inds, :]),
                    (bin_sizes[i], bin_sizes[i]),
                    weights=weights_tmp
                )[0]
    MIs = np.stack([
        MI_samples.mean(axis=-1),
        MI_samples.std(axis=-1),
    ], axis=-1)
    return MIs


def direct_method(
    x, y, bin_sizes, fractions, num_samples,
    weights=None,
    shuffle_threshold=0.02, return_slope=False,
    return_for_graph=False
):
    if not x.shape[0] == y.shape[0]:
        raise ValueError("x and y must have same length in first dimension.")
    if y.ndim == 1:
        MIs = create_samples_1d(
            x, y, bin_sizes, fractions, num_samples, weights=weights
        )
    else:
        MIs = create_samples_1d_dd(
            x, y, bin_sizes, fractions, num_samples, weights=weights
        )
    MIs = MIs.reshape((-1, len(fractions), 2))

    params = fit_linear_to_array(MIs[..., 0], 1/fractions, MIs[..., 1])
    params = params.reshape((len(bin_sizes), 2, 4))

    where_less = params[:, 1, 2] < shuffle_threshold
    if where_less.sum() < 3:
        where_less = np.zeros(params.shape[0], dtype=bool)
        where_less[:3] = True

    model = LinearModel()
    result = model.fit(
        data=params[where_less, 0, 2],
        x=1/bin_sizes[where_less]**2,
        weights=params[where_less, 0, 3]**2,
    )
    mi = result.params["intercept"].value
    mi_err = result.params["intercept"].stderr
    if not return_for_graph:
        if return_slope:
            return mi, mi_err, result.params["slope"].value
        else:
            return mi, mi_err
    else:
        return (
            mi, mi_err,
            MIs.reshape(len(bin_sizes), 2, len(fractions), 2), params,
            result.params["slope"].value, where_less
        )


def graph_direct_method(
    x, y, bin_sizes, fractions, num_samples,
    weights=None,
    shuffle_threshold=0.02
):
    mi, mi_err, MIs, params, mi_slope, where_less = direct_method(
        x, y, bin_sizes, fractions, num_samples, weights, shuffle_threshold,
        return_for_graph=True
    )
    fig, axes = plt.subplots(1, 3, figsize=[30, 10])
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    for i in range(len(bin_sizes)):
        errbarcont = axes[0].errorbar(
            1/fractions, MIs[i, 0, :, 0],
            yerr=MIs[i, 0, :, 1],
            linestyle="none",  # markersize=10,
            marker="x",
        )
        color = errbarcont.lines[0].get_color()
        x = np.zeros((len(fractions)+1))
        x[1:] = 1/fractions
        y = params[i, 0, 0]*x + params[i, 0, 2]
        axes[0].plot(x, y, color=color)
        axes[0].errorbar(
            [0], [params[i, 0, 2]], yerr=[params[i, 0, 3]],
            color="red", marker="x"
        )

    axes[0].set_xlabel("1/(# samples)")
    axes[0].set_ylabel("I [bits]")

    labels_bins = [None, "randomly shuffled"]
    for j in range(2):
        axes[1].errorbar(
            bin_sizes, params[:, j, 2], yerr=params[:, j, 3],
            linestyle="-", marker="x",
            label=labels_bins[j],
            color="red" if j == 0 else None
        )
    axes[1].axhline(
        shuffle_threshold, linestyle="--",
        label="threshold at {} bits".format(shuffle_threshold)
    )
    axes[1].set_xlabel("# bins")
    axes[1].set_ylabel("I [bits]")
    axes[1].legend()

    axes[2].errorbar(
        1/bin_sizes[where_less]**2, params[where_less, 0, 2],
        yerr=params[where_less, 0, 3],
        linestyle="none", marker="x", color="red"
    )
    x = np.asarray([0, 1e-2])
    y = x*mi_slope + mi
    axes[2].plot(
        x, y
    )
    axes[2].errorbar([0], [mi], yerr=[mi_err], color="green", marker="x")

    axes[2].set_xlabel("1/(# bins)^2")
    axes[2].set_ylabel("I [bits]")

    plt.show()
