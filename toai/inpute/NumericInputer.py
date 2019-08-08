from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericInputer(BaseEstimator, TransformerMixin):
    IMPLEMENTED_STRATEGIES = {"median": np.nanmedian, "mean": np.nanmean}

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "median",
        suffix: str = "_na",
    ):
        if strategy not in self.IMPLEMENTED_STRATEGIES:
            raise NotImplementedError("This strategy is not implemented.")
        self.strategy = strategy
        self.columns = columns
        self.suffix = suffix

    def fit(self, data: pd.DataFrame) -> "NumericInputer":
        self.statistics_ = {
            column: self.IMPLEMENTED_STRATEGIES[self.strategy](data[column])
            for column in self.columns or data.columns
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column in self.statistics_:
            data.insert(
                data.columns.get_loc(column) + 1,
                column + self.suffix,
                data[column].isna().astype(int),
            )
            data[column].fillna(value=self.statistics_[column], inplace=True)
        return data
