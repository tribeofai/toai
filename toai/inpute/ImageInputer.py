import os
from typing import List, Optional, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ImageInputer(BaseEstimator, TransformerMixin):
    def __init__(self, value: Any, columns: Optional[List[str]] = None):
        self.value = value
        self.columns = columns

    def fit(self, data: pd.DataFrame) -> "ImageInputer":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        def fill_missing_with_value(value):
            def inner(x):
                return x if os.path.isfile(x) else value

            return inner

        data = data.copy()
        for column in self.columns or data.columns:
            data[column] = (
                data[column].apply(fill_missing_with_value(self.value)).values
            )
        return data
