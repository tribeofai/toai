import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from ..data import DataBundle
from ..utils import save_file, load_file


class ImageDataBundle(DataBundle):
    @classmethod
    def from_subfolders(cls, path: Union[Path, str]) -> "ImageDataBundle":
        path = Path(path)
        paths = []
        labels = []
        for label in os.listdir(path):
            for image_path in os.listdir(path / label):
                paths.append(str(path / label / image_path))
                labels.append(label)

        return cls(np.asarray(paths), np.asarray(labels))

    @classmethod
    def from_re(
        cls,
        path: Union[Path, str],
        regex: str,
        default: Optional[Union[int, float, str, bool]] = None,
    ) -> "ImageDataBundle":
        paths = []
        labels = []
        for value in os.listdir(path):
            match = re.match(regex, value)
            if match:
                labels.append(match.group(1))
            elif default:
                labels.append(str(default))
            else:
                raise ValueError(
                    f"No match found and no default value provided for value: {value}"
                )
            paths.append(f"{path}/{value}")

        return cls(np.asarray(paths), np.asarray(labels))

    def dataset(
        self,
        batch_size: int,
        img_dims: Tuple[int, int, int],
        shuffle: bool = False,
        prefetch: int = tf.data.experimental.AUTOTUNE,
        n_parallel_calls: int = -1,
    ):
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.n_parallel_calls = n_parallel_calls
        self.steps = math.ceil(len(self) / self.batch_size)
        return self

    def make_label_map(self) -> Dict[str, int]:
        return {value: key for key, value in dict(enumerate(np.unique(self.y))).items()}

    def make_label_scaler(self) -> Any:
        label_scaler = sklearn.preprocessing.RobustScaler()
        label_scaler.fit(self.y)
        return label_scaler

    def make_pipeline(
        self,
        regression: bool = False,
        label_map: Optional[Dict[Union[str, int], int]] = None,
        label_scaler: Optional[sklearn.base.BaseEstimator] = None,
        image_pipeline: Optional[List[Callable]] = None,
    ) -> "ImageDataBundle":
        self.regression = regression
        if self.regression:
            self.label_scaler = label_scaler or self.make_label_scaler()
        else:
            self.label_map = label_map or self.make_label_map()
            self.classes = list(self.label_map.keys())
            self.n_classes = len(self.classes)

        self.image_pipeline = image_pipeline or []
        return self

    def load_pipeline(
        self, path: Union[Path, str], regression: bool = False
    ) -> "ImageDataBundle":
        path = Path(path)
        self.regression = regression
        if self.regression:
            self.label_scaler = load_file(path / "label_scaler.pickle")
        else:
            self.label_map = load_file(path / "label_map.pickle")
            self.classes = list(self.label_map.keys())
            self.n_classes = len(self.classes)

        self.image_pipeline = load_file(path / "image_pipeline.pickle")
        return self

    def save_pipeline(self, path: Union[Path, str]) -> "ImageDataBundle":
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.regression:
            save_file(self.label_scaler, path / "label_scaler.pickle")
        else:
            save_file(self.label_map, path / "label_map.pickle")

        save_file(self.image_pipeline, path / "image_pipeline.pickle")
        return self

    def preprocess_with_pipeline(
        self, data_bundle: tf.data.Dataset, pipeline: List[Callable]
    ) -> tf.data.Dataset:
        for fun in pipeline:
            data_bundle = data_bundle.map(fun, num_parallel_calls=self.n_parallel_calls)
        return data_bundle

    def preprocess(self) -> "ImageDataBundle":
        if self.regression:
            label_ds = tf.data.Dataset.from_tensor_slices(
                self.label_scaler.transform(self.y)
            )
        else:
            label_ds = tf.data.Dataset.from_tensor_slices(
                np.asarray([self.label_map[label] for label in self.y])
            )

        image_ds = tf.data.Dataset.from_tensor_slices(self.x)

        image_ds = self.preprocess_with_pipeline(image_ds, self.image_pipeline)

        data_bundle = tf.data.Dataset.zip((image_ds, label_ds))
        if self.shuffle:
            data_bundle = data_bundle.shuffle(len(self))
        self.data = data_bundle.repeat().batch(self.batch_size).prefetch(self.prefetch)
        return self

    def show(self, cols: int = 8, n_batches: int = 1, debug: bool = False):
        if cols >= self.batch_size * n_batches:
            cols = self.batch_size * n_batches
            rows = 1
        else:
            rows = math.ceil(self.batch_size * n_batches / cols)

        figsize = (3 * cols, 4 * rows) if debug else (3 * cols, 3 * rows)
        _, ax = plt.subplots(rows, cols, figsize=figsize)

        i = 0
        for x_batch, y_batch in self.data.take(n_batches):
            for (x, y) in zip(x_batch.numpy(), y_batch.numpy()):
                idx = (i // cols, i % cols) if rows > 1 else i % cols
                ax[idx].axis("off")
                ax[idx].imshow(x)
                title = f"Label: {y}\nShape: {x.shape}\n" if debug else y
                ax[idx].set_title(title)
                i += 1
