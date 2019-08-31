from pathlib import Path
from typing import Union
from tensorflow import keras


def load_keras_model(
    architecture_path: Union[Path, str], weights_path: Union[Path, str]
):
    with open(architecture_path, "r") as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    return model
