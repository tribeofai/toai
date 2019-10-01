import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def split(
        cls,
        dataset: "Dataset",
        fracs: Union[List[float], Tuple[float]],
        random: Optional[bool] = True,
    ) -> Tuple["Dataset"]:
        x = dataset.x
        y = dataset.y

        if random:
            random_indices = np.random.permutation(len(dataset))
            x = x[random_indices]
            y = y[random_indices]

        result = []
        current_index = 0
        for frac in fracs:
            dx = math.ceil(len(dataset) * frac)
            split_dataset = cls(
                x=x[current_index : current_index + dx],
                y=y[current_index : current_index + dx],
            )
            result.append(split_dataset)
            current_index += dx

        return tuple(result)

    @classmethod
    def from_dataframe(
        cls, dataframe: pd.DataFrame, x_col: str, y_col: str
    ) -> "Dataset":
        return cls(dataframe[x_col].values, dataframe[y_col].values)

    def __len__(self) -> int:
        return len(self.y)
