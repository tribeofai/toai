from sklearn.metrics import accuracy_score


def error_rate(y_true, y_pred, normalize=True, sample_weight=None):
    return 1 - accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
