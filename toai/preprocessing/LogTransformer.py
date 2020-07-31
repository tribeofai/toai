from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns: Optional[List[str]] = None,
    ):
        self.columns = columns

    def fit(self, data: pd.DataFrame) -> "LogTransformer":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column in self.columns or data.columns:
            data[column] = data[column].apply(np.log1p).values
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column in self.columns or data.columns:
            data[column] = data[column].apply(np.expm1).values
        return data
