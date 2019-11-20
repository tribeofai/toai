import math
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def from_unbalanced_data_bundle(cls, data_bundle: "DataBundle") -> "DataBundle":
        values, counts = np.unique(data_bundle.y, return_counts=True)
        max_count = counts.max()
        x = []
        y = []
        for value, count in zip(values, counts):
            indices = np.argwhere(data_bundle.y == value)
            for _ in range(max_count // count):
                x.append(data_bundle.x[indices].flatten())
                y.append(data_bundle.y[indices].flatten())

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

    def value_counts(self) -> Dict[str, int]:
        values, counts = np.unique(self.y, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))
