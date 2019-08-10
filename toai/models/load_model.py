from tensorflow import keras


def load_keras_model(architecture_path: str, weights_path: str):
    with open(architecture_path, "r") as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    return model
