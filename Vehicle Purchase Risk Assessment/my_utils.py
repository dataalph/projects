import pandas as pd
import numpy as np

def time_ordered_split_3(
    X, y, date_field, train_size=1 / 3, valid_size=1 / 3, test_size=1 / 3
):
    """
    Time-ordered split into train / validation / test based on date field.
    Data is sorted by date and split sequentially.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    date_field : str
        Name of the date column to sort by
    train_size : float, default 1/3
        Proportion of data for training (earliest dates)
    valid_size : float, default 1/3
        Proportion of data for validation (middle dates)
    test_size : float, default 1/3
        Proportion of data for testing (latest dates)

    Returns
    -------
    X_train, X_valid, X_test,
    y_train, y_valid, y_test
    """

    # Validate proportions
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    if train_size <= 0 or valid_size <= 0 or test_size <= 0:
        raise ValueError("All size parameters must be positive")

    X = X.copy()
    X[date_field] = pd.to_datetime(X[date_field])

    # Sort by date (earliest first)
    X = X.sort_values(date_field)
    y = y.loc[X.index]

    n_samples = len(X)

    # Calculate split indices
    train_end = int(n_samples * train_size)
    valid_end = int(n_samples * (train_size + valid_size))

    # Sequential split (no shuffling)
    X_train = X.iloc[:train_end]
    X_valid = X.iloc[train_end:valid_end]
    X_test = X.iloc[valid_end:]

    y_train = y.iloc[:train_end]
    y_valid = y.iloc[train_end:valid_end]
    y_test = y.iloc[valid_end:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
