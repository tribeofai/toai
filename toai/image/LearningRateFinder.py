import tempfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List


class LearningRateFinder:
    def __init__(self, model: keras.Model, stop_factor: int = 4, beta: float = 0.98):
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.lr_multiplier = 1.0
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num = 0
        self.weights_path: Optional[str] = None

    def reset(self):
        self.lrs = []
        self.losses = []
        self.lr_multiplier = 1.0
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num = 0
        self.weights_path = None

    def on_batch_end(self, batch, logs):
        lr = self.model.optimizer.lr
        self.lrs.append(lr.numpy())
        loss = logs["loss"]
        self.batch_num += 1
        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * loss)
        smooth = self.avg_loss / (1 - (self.beta ** self.batch_num))
        self.losses.append(smooth)

        stop_loss = self.stop_factor * self.best_loss

        if self.batch_num > 1 and smooth > stop_loss:
            self.model.stop_training = True
            return

        if self.batch_num == 1 or smooth < self.best_loss:
            self.best_loss = smooth

        lr = lr * self.lr_multiplier
        self.model.optimizer.lr = lr

    def find(
        self,
        x: tf.data.Dataset,
        start_lr: float,
        end_lr: float,
        epochs: Optional[int] = None,
        steps_per_epoch: int = 1,
        batch_size: int = 32,
        sample_size: int = 2048,
        verbose: int = 1,
    ):
        self.reset()
        if epochs is None:
            epochs = int(np.ceil(sample_size / steps_per_epoch))

        num_batch_updates = epochs * steps_per_epoch

        self.lr_multiplier = (end_lr / start_lr) ** (1.0 / num_batch_updates)

        self.weights_path = tempfile.mkstemp()[1]
        self.model.save_weights(self.weights_path)

        original_lr = self.model.optimizer.lr.numpy()
        self.model.optimizer.lr = start_lr

        self.model.fit(
            x,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[keras.callbacks.LambdaCallback(on_batch_end=self.on_batch_end)],
            verbose=verbose,
        )

        self.model.load_weights(self.weights_path)
        self.model.optimizer.lr = original_lr

    def plot_loss(
        self, skip_begin: int = 10, skip_end: int = 1, title: Optional[str] = None
    ):
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]

        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        if title:
            plt.title(title)
