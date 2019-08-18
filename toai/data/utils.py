from sklearn.model_selection import train_test_split


def split_df(data, test_size, target_col=None, random_state=42):
    if target_col:
        stratify = data[target_col]
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=stratify, random_state=random_state
    )
    if target_col:
        test_stratify = test_data[target_col]
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, stratify=test_stratify, random_state=random_state
    )
    for df in train_data, val_data, test_data:
        df.reset_index(drop=True, inplace=True)
    return train_data, val_data, test_data
