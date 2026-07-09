import pandas as pd
import numpy as np

def time_ordered_split_3_new(
    X, y, date_field, train_size=1 / 3, valid_size=1 / 3, test_size=1 / 3
):
    """
    Time-ordered split into train / validation / test based on date field.
    Data is sorted by date and split sequentially, ensuring that all records
    with the same date are in the same split.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    date_field : str
        Name of the date column to sort by
    train_size : float, default 1/3
        Approximate proportion of data for training (earliest dates)
    valid_size : float, default 1/3
        Approximate proportion of data for validation (middle dates)
    test_size : float, default 1/3
        Approximate proportion of data for testing (latest dates)

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

    # Get unique dates in order
    unique_dates = X[date_field].unique()
    n_unique_dates = len(unique_dates)

    # Calculate split points based on unique dates
    train_dates_end = int(n_unique_dates * train_size)
    valid_dates_end = int(n_unique_dates * (train_size + valid_size))

    # Ensure at least one date in each split
    train_dates_end = max(1, train_dates_end)
    valid_dates_end = max(train_dates_end + 1, valid_dates_end)
    valid_dates_end = min(valid_dates_end, n_unique_dates - 1)

    # Get the actual date boundaries
    train_end_date = unique_dates[train_dates_end - 1]
    valid_end_date = unique_dates[valid_dates_end - 1]

    # Split based on date boundaries
    X_train = X[X[date_field] <= train_end_date]
    X_valid = X[(X[date_field] > train_end_date) & (X[date_field] <= valid_end_date)]
    X_test = X[X[date_field] > valid_end_date]

    # Get corresponding y values
    y_train = y.loc[X_train.index]
    y_valid = y.loc[X_valid.index]
    y_test = y.loc[X_test.index]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def update_model_results(result_df, model_name, data_type, gini_score):
    """
    Appends a Gini score record to the model results dataframe.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Dataframe with model results
    model_name : str
        Name of the model
    data_type : str
        Data type ('train', 'valid', 'test')
    gini_score : float
        Gini coefficient value
    
    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    
    result_df = pd.concat(
        [
            result_df,
            pd.DataFrame([{
                "model": model_name, 
                "data_type": data_type, 
                "gini": gini_score
            }])
        ],
        ignore_index=True
    )
    
    return result_df