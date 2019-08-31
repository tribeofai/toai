from pathlib import Path
from typing import Union

from tensorflow import keras


def save_keras_model(
    model: keras.Model,
    architecture_path: Union[Path, str],
    weights_path: Union[Path, str],
):
    model.save_weights(weights_path)
    with open(architecture_path, "w") as f:
        f.write(model.to_json())
