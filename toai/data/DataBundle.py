import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def split(
        cls,
        data_bundle: "DataBundle",
        fracs: Union[List[float], Tuple[float]],
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
