from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Dict


class Extractor(BaseEstimator, TransformerMixin):
    def __init__(
        self, source_column: str, patterns: Dict[str, str], drop_source: bool = False
    ):
        self.patterns = patterns
        self.source_column = source_column
        self.drop_source = drop_source

    def fit(self, data: pd.DataFrame) -> "Extractor":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        col_index = data.columns.get_loc(self.source_column)
        for i, (column, pattern) in enumerate(self.patterns.items(), 1):
            data.insert(
                col_index + i, column, data[self.source_column].str.extract(pattern)
            )
        if self.drop_source:
            data.drop(self.source_column, axis=1)
        return data
