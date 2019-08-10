from tensorflow import keras


def save_keras_model(model: keras.Model, architecture_path: str, weights_path: str):
    model.save_weights(weights_path)
    with open(architecture_path, "w") as f:
        f.write(model.to_json())
