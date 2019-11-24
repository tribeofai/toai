import math
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def from_unbalanced(
        cls,
        data_bundle: "DataBundle",
        target_class_size: int,
        value_counts: Dict[int, int],
    ) -> "DataBundle":
        x = []
        y = []
        for label, current_size in value_counts.items():
            current_indices = np.argwhere(data_bundle.y == label).flatten()
            next_indices = []
            for _ in range(target_class_size // current_size):
                next_indices.append(np.random.permutation(current_indices))
            next_indices.append(
                np.random.choice(current_indices, target_class_size % current_size)
            )
            next_indices = np.concatenate(next_indices)
            x.append(data_bundle.x[next_indices])
            y.append(data_bundle.y[next_indices])

        return cls(np.concatenate(x), np.concatenate(y))

    @classmethod
    def split(
        cls,
        data_bundle: "DataBundle",
        fracs: Sequence[float],
        random: Optional[bool] = True,
    ) -> Tuple["DataBundle", ...]:
        x = data_bundle.x
        y = data_bundle.y

        if random:
            random_indices = np.random.permutation(len(data_bundle))
            x = x[random_indices]
            y = y[random_indices]

        result = []
        current_index = 0
        for frac in fracs:
            dx = math.ceil(len(data_bundle) * frac)
            split_data_bundle = cls(
                x=x[current_index : current_index + dx],
                y=y[current_index : current_index + dx],
            )
            result.append(split_data_bundle)
            current_index += dx

        return tuple(result)

    @classmethod
    def from_dataframe(
        cls, dataframe: pd.DataFrame, x_col: str, y_col: str
    ) -> "DataBundle":
        return cls(dataframe[x_col].values, dataframe[y_col].values)

    def __len__(self) -> int:
        return len(self.y)

    def to_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x, self.y))

    def value_counts(self) -> Dict[str, int]:
        values, counts = np.unique(self.y, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    def make_label_map(self) -> Dict[str, int]:
        return {value: key for key, value in enumerate(np.unique(self.y))}

    def apply_label_map(self, label_map: Dict[str, int]) -> None:
        self.y = np.vectorize(label_map.get)(self.y)

    def make_label_scaler(self) -> sklearn.base.BaseEstimator:
        scaler = sklearn.preprocessing.RobustScaler()
        scaler.fit(self.y)
        return scaler
