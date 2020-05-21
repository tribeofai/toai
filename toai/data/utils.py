from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def split_df(
    data: pd.DataFrame,
    test_size: float,
    target_col: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = data[target_col] if target_col else None
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=stratify, random_state=random_state
    )
    test_stratify = test_data[target_col] if target_col else None
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, stratify=test_stratify, random_state=random_state
    )
    for df in train_data, val_data, test_data:
        df.reset_index(drop=True, inplace=True)
    return train_data, val_data, test_data


def balance_df_labels(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    max_value = df[col_name].value_counts().max()
    result = [
        df[df[col_name] == value]
        for value, count in df[col_name].value_counts().items()
        for _ in range(max_value // count)
    ]
    return pd.concat(result).reset_index(drop=True)
