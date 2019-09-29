import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..data import Dataset
from ..models import load_keras_model, save_keras_model


class ImageLearner:
    def __init__(
        self,
        path,
        base_model,
        input_shape,
        output_shape,
        activation,
        loss,
        metrics,
        dropout=0.0,
        l1=None,
        l2=None,
        override=False,
        load=False,
    ):
        self.path = str(path)
        self.weights_path = f"{self.path}/weights.h5"
        self.architecture_path = f"{self.path}/model.json"
        self.logs_path = f"{self.path}/logs"

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2

        self.base_model = base_model(include_top=False, input_shape=input_shape)
        x = keras.layers.concatenate(
            [
                keras.layers.GlobalAvgPool2D()(self.base_model.output),
                keras.layers.GlobalMaxPool2D()(self.base_model.output),
            ]
        )
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)

        if self.l1 is not None and self.l2 is not None:
            kernel_regularizer = keras.regularizers.l1_l2(self.l1, self.l2)
        elif self.l1 is None and self.l2 is not None:
            kernel_regularizer = keras.regularizers.l2(self.l2)
        elif self.l1 is not None and self.l2 is None:
            kernel_regularizer = keras.regularizers.l1(self.l1)
        else:
            kernel_regularizer = None

        outputs = [
            keras.layers.Dense(
                output_size,
                kernel_regularizer=kernel_regularizer,
                activation=activation,
            )(x)
            for output_size in self.output_shape
        ]

        self.model = keras.Model(inputs=self.base_model.inputs, outputs=outputs)

        if os.path.exists(self.path):
            if load:
                self.load()
            elif override:
                shutil.rmtree(self.path)
                os.makedirs(self.path)
        else:
            os.makedirs(self.path)

        self.save()

    def save(self):
        save_keras_model(self.model, self.architecture_path, self.weights_path)

    def load(self, weights_only: bool = False):
        if weights_only:
            self.model.load_weights(self.weights_path)
        else:
            self.model = load_keras_model(self.architecture_path, self.weights_path)

    def compile(self, optimizer: keras.optimizers, lr: float):
        self.model.compile(
            optimizer=optimizer(lr), loss=self.loss, metrics=self.metrics
        )

    def freeze(self, n_layers: int = 1):
        for layer in self.model.layers[:-n_layers]:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.model.layers:
            layer.trainable = True

    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        epochs: int,
        verbose: int = 1,
    ):
        reduce_lr_patience = max(2, epochs // 4)
        early_stopping_patience = reduce_lr_patience * 2

        self.history = self.model.fit(
            x=train_dataset.data,
            steps_per_epoch=train_dataset.steps,
            validation_data=validation_dataset.data,
            validation_steps=validation_dataset.steps,
            epochs=epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    self.weights_path, save_best_only=True, save_weights_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.3, patience=reduce_lr_patience
                ),
                keras.callbacks.EarlyStopping(
                    patience=early_stopping_patience, restore_best_weights=True
                ),
            ],
            verbose=verbose,
        )
        self.load(weights_only=True)

    def predict(self, path: Optional[str] = None, image=None):
        if image is None:
            image = tf.data.Dataset.from_tensor_slices([path])
            image = self.data.test.preprocess(image, 1).batch(1)
        elif image.ndim == 3:
            image = image[np.newaxis, :]
        return self.model.predict(image)

    def show_history(self, contains: str, skip: int = 0):
        history_df = pd.DataFrame(self.history.history)
        history_df[list(history_df.filter(regex=contains))].iloc[skip:].plot()
