import numpy as np
from lmfit.models import LinearModel


def fit_linear_to_array(data, x, data_err):
    """
    Runs the same linear fit for multiple data sets.
    data: Array of shape (samples, N). Where samples is the number of fits to
        run and N is the number of datapoints per fit.
    x: Array of shape (N,). The x values for the fit.
    data_err: Array of shape (samples, N). The std values for data.
    Returns params:
        Array of shape (samples, 4), where params[i, j] is the slope (j=0) of
        data[i, :], the slope standard deviation (j=1), intercept (j=2),
        intercept standard deviation (j=3).
    """
    params = np.empty((data.shape[0], 4))
    model = LinearModel()
    for i in range(data.shape[0]):
        result = model.fit(data=data[i, :], x=x, weights=1/data_err[i, :]**2)

        params[i, 0] = result.params["slope"].value
        params[i, 1] = result.params["slope"].stderr
        params[i, 2] = result.params["intercept"].value
        params[i, 3] = result.params["intercept"].stderr
    return params
