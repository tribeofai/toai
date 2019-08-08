from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalInputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, suffix: str = "_na"):
        self.columns = columns
        self.suffix = suffix

    def fit(self, data: pd.DataFrame) -> "CategoricalInputer":
        self.statistics_ = {
            column: data[column].mode().values[0]
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
