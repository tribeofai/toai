from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Adder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        col_name: str,
        columns: Optional[List[str]] = None,
        drop_source: bool = False,
    ):
        self.col_name = col_name
        self.columns = columns
        self.drop_source = drop_source

    def fit(self, data) -> "Adder":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if not self.columns:
            self.columns = data.columns
        data.insert(len(self.columns), self.col_name, data[self.columns].sum(axis=1))

        if self.drop_source:
            for column in self.columns:
                data.drop(column, inplace=True, axis=1)
        return data
