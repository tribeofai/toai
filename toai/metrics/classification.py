from tensorflow import keras


def sparse_top_2_categorical_accuracy(*args, **kwargs):
    return keras.metrics.sparse_top_k_categorical_accuracy(*args, **kwargs, k=2)
