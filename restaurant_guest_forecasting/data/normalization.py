import pandas as pd

from typing import Tuple
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.data.normalizer.normalizer import Normalizer


def normalize_features_and_targets(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    is_train: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalizes features and targets using the Normalizer class.
    Handles fitting and loading for train/test splits.
    Returns normalized X and y as DataFrames.
    """
    X_normalizer = Normalizer(is_target=False)
    y_normalizer = Normalizer(is_target=True)

    if is_train:
        X_normalizer.fit(X_df, save=True)
        X = X_normalizer.transform(X_df)

        y_normalizer.fit(y_df, save=True)
        y = y_normalizer.transform(y_df)
    else:
        X_normalizer.load()
        y_normalizer.load()

        X = X_normalizer.transform(X_df)
        y = y_normalizer.transform(y_df)

    return X, y

def preprocess_df(
    df: pd.DataFrame,
    target_column: str = "GUESTS",
    art_prefix: str = "art_"
) -> pd.DataFrame:
    """
    Preprocesses a DataFrame for guest prediction.
    Splits the date, drops article columns, and normalizes features.
    Returns the processed DataFrame.
    """
    
    df = split_date(df, drop_date=True)

    art_columns = [col for col in df.columns if col.startswith(art_prefix)]
    df = df.drop(columns=art_columns)

    X_df = df.drop(columns=[target_column])
    y_df = df[[target_column]]

    return X_df, y_df


def normalize_df(
    df: pd.DataFrame,
    is_train: bool = True,
    target_column: str = "GUESTS",
    art_prefix: str = "art_"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalizes a DataFrame for guest prediction.
    Splits the date, drops article columns, and normalizes features and target.
    """    

    X_df, y_df = preprocess_df(df, target_column, art_prefix)

    X, y = normalize_features_and_targets(X_df, y_df, is_train)

    return X, y