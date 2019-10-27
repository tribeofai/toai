import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..models import load_keras_model, save_keras_model
from .ImageDataBundle import ImageDataBundle


class ImageLearner:
    def __init__(
        self,
        path: Union[Path, str],
        base_model: keras.Model,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        activation: Union[str, Callable],
        loss: Union[str, Callable],
        metrics: List[Callable],
        dropout: float = 0.0,
        l1: Optional[float] = None,
        l2: Optional[float] = None,
        override: bool = False,
        load: bool = False,
        class_weight: Optional[Dict[int, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
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
        self.class_weight = class_weight
        self.sample_weight = sample_weight

        self.base_model = base_model(include_top=False, input_shape=input_shape)
        self.concat_layer = keras.layers.concatenate(
            [
                keras.layers.GlobalAvgPool2D()(self.base_model.output),
                keras.layers.GlobalMaxPool2D()(self.base_model.output),
            ]
        )
        self.bn_layer = keras.layers.BatchNormalization()(self.concat_layer)
        self.dropout_layer = keras.layers.Dropout(dropout)(self.bn_layer)

        if self.l1 is not None and self.l2 is not None:
            kernel_regularizer = keras.regularizers.l1_l2(self.l1, self.l2)
        elif self.l1 is None and self.l2 is not None:
            kernel_regularizer = keras.regularizers.l2(self.l2)
        elif self.l1 is not None and self.l2 is None:
            kernel_regularizer = keras.regularizers.l1(self.l1)
        else:
            kernel_regularizer = None

        self.output_layers = [
            keras.layers.Dense(
                output_size,
                kernel_regularizer=kernel_regularizer,
                activation=activation,
            )(self.dropout_layer)
            for output_size in self.output_shape
        ]

        self.model = keras.Model(
            inputs=self.base_model.inputs, outputs=self.output_layers
        )

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

    def compile(self, optimizer: keras.optimizers.Optimizer, lr: float):
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
        train_data_bundle: ImageDataBundle,
        validation_data_bundle: ImageDataBundle,
        epochs: int,
        verbose: int = 1,
    ):
        reduce_lr_patience = max(2, epochs // 4)
        early_stopping_patience = reduce_lr_patience * 2

        self.history = self.model.fit(
            x=train_data_bundle.data,
            steps_per_epoch=train_data_bundle.steps,
            validation_data=validation_data_bundle.data,
            validation_steps=validation_data_bundle.steps,
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
            class_weight=self.class_weight,
            sample_weight=self.sample_weight,
        )
        self.load(weights_only=True)

    def predict(
        self,
        pipeline: List[Callable],
        image_paths: Optional[Iterable[str]] = None,
        images: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if images is None:
            images = tf.data.Dataset.from_tensor_slices(image_paths)
        for fun in pipeline:
            images = images.map(fun, num_parallel_calls=1)
        images = images.batch(1)
        return self.model.predict(images)

    def show_history(self, contains: str, skip: int = 0):
        history_df = pd.DataFrame(self.history.history)
        history_df[list(history_df.filter(regex=contains))].iloc[skip:].plot()
