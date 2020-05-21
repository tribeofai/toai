from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score


def error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    return 1 - accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
