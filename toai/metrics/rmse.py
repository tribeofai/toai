import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    return np.sqrt(
        mean_squared_error(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
    )
