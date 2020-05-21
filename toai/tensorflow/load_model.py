from pathlib import Path
from typing import Union, Optional
from tensorflow import keras


def load_keras_model(
    architecture_path: Union[Path, str],
    weights_path: Union[Path, str],
    custom_objects: Optional[dict] = None,
):
    with open(architecture_path, "r") as f:
        model = keras.models.model_from_json(f.read(), custom_objects=custom_objects)
    model.load_weights(weights_path)
    return model
