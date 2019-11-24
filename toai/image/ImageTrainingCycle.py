import attr

import tensorflow as tf
from tensorflow import keras


@attr.s(auto_attribs=True)
class ImageTrainingCycle:
    data: tf.data.Dataset
    steps: int
    n_epochs: int
    lr: float
    optimizer: keras.optimizers.Optimizer
    freeze: bool = False
