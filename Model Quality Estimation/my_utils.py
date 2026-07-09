import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def clean_symbols(text):
    return re.sub(r"[\[\]\'\" ]", "", str(text))


def process_features_complete(df_):
    """
    Fully processes the 'features' column in training data by extracting binary features
    from the list of amenities. The function performs text cleaning, tokenization,
    frequency counting of unique amenities, and creates 20 binary features based on the
    most popular amenities. Finally adds 'bathrooms' and 'bedrooms' features for modeling.

    Parameters
    ----------
    df_ : pandas.DataFrame
        Input DataFrame containing at least 'features', 'bathrooms', 'bedrooms', and 'price' columns.

    Returns
    -------
    df : pandas.DataFrame
        Feature matrix with binary amenity features plus 'bathrooms', 'bedrooms', 'created' and 'price'.
    """

    df = df_.copy()

    df["features_clean"] = df["features"].apply(clean_symbols)

    df["features_split"] = df["features_clean"].str.split(",")

    all_features = []
    for feature_list in df["features_split"]:
        if feature_list:
            all_features.extend(feature_list)

    all_features = [f for f in all_features if f != ""]

    feature_counts = Counter(all_features)
    top_20_features = [feature for feature, count in feature_counts.most_common(20)]

    for feature in top_20_features:
        df[f"{feature}"] = df["features_split"].apply(
            lambda lst: 1 if feature in lst else 0
        )

    binary_features = [f"{f}" for f in top_20_features]
    feature_list = binary_features + ["bathrooms", "bedrooms", "created", "price"]

    return df[feature_list]


def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def add_model_results(
    result_MAE,
    result_RMSE,
    result_R2,
    model,
    model_name,
    X_train_,
    X_test_,
    y_train,
    y_test,
):
    """
    Trains a model and adds evaluation results to results tables
    """

    # Training
    model.fit(X_train_, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_)
    y_test_pred = model.predict(X_test_)

    # MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # R2
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Adding rows
    result_MAE = pd.concat(
        [
            result_MAE,
            pd.DataFrame([{"model": model_name, "train": train_mae, "test": test_mae}]),
        ],
        ignore_index=True,
    )

    result_RMSE = pd.concat(
        [
            result_RMSE,
            pd.DataFrame(
                [{"model": model_name, "train": train_rmse, "test": test_rmse}]
            ),
        ],
        ignore_index=True,
    )

    result_R2 = pd.concat(
        [
            result_R2,
            pd.DataFrame([{"model": model_name, "train": train_r2, "test": test_r2}]),
        ],
        ignore_index=True,
    )

    return result_MAE, result_RMSE, result_R2


def add_model_results_cv(
    result_MAE, result_RMSE, result_R2, model, model_name, X, y, folds
):
    """
    Trains model using cross-validation folds
    and adds averaged metrics to results tables
    """

    train_mae_list = []
    test_mae_list = []

    train_rmse_list = []
    test_rmse_list = []

    train_r2_list = []
    test_r2_list = []

    for train_idx, test_idx in folds:

        X_train_, X_test_ = X.iloc[train_idx], X.iloc[test_idx]
        y_train_, y_test_ = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_, y_train_)

        y_train_pred = model.predict(X_train_)
        y_test_pred = model.predict(X_test_)

        train_mae_list.append(mean_absolute_error(y_train_, y_train_pred))
        test_mae_list.append(mean_absolute_error(y_test_, y_test_pred))

        train_rmse_list.append(np.sqrt(mean_squared_error(y_train_, y_train_pred)))
        test_rmse_list.append(np.sqrt(mean_squared_error(y_test_, y_test_pred)))

        train_r2_list.append(r2_score(y_train_, y_train_pred))
        test_r2_list.append(r2_score(y_test_, y_test_pred))

    train_mae = np.mean(train_mae_list)
    test_mae = np.mean(test_mae_list)

    train_rmse = np.mean(train_rmse_list)
    test_rmse = np.mean(test_rmse_list)

    train_r2 = np.mean(train_r2_list)
    test_r2 = np.mean(test_r2_list)

    result_MAE = pd.concat(
        [
            result_MAE,
            pd.DataFrame([{"model": model_name, "train": train_mae, "test": test_mae}]),
        ],
        ignore_index=True,
    )

    result_RMSE = pd.concat(
        [
            result_RMSE,
            pd.DataFrame(
                [{"model": model_name, "train": train_rmse, "test": test_rmse}]
            ),
        ],
        ignore_index=True,
    )

    result_R2 = pd.concat(
        [
            result_R2,
            pd.DataFrame([{"model": model_name, "train": train_r2, "test": test_r2}]),
        ],
        ignore_index=True,
    )

    return result_MAE, result_RMSE, result_R2
