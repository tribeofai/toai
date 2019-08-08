from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def fit(self, data: pd.DataFrame) -> "CategoricalEncoder":
        data = data.astype("category")
        self.categories_ = {
            column: dict(enumerate(data[column].cat.categories, 1))
            for column in self.columns or data.columns
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column in self.categories_:
            data[column] = (
                data[column]
                .map({value: key for key, value in self.categories_[column].items()})
                .fillna(value=0)
                .astype(int)
            )
        return data
