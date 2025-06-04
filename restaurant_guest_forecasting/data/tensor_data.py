import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from typing import Callable
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.data.normalization import normalize_features_and_targets



def guest_df_to_tensor_dataset(df: pd.DataFrame, is_train: bool = True) -> TensorDataset:
    """
    Converts a DataFrame into a PyTorch TensorDataset for guest prediction.
    Also normalizes the features and target.
    Assumes:
    - All columns are numeric.
    - 'GUESTS' is the target column.
    - Columns starting with 'art_' are additional features to be dropped.
    Returns:
        TensorDataset: Features (X) and target (y).
    """ 
    art_columns = [col for col in df.columns if col.startswith("art_")]
    df = df.drop(columns=art_columns)

    X_df = df.drop(columns=["GUESTS"])
    y_df = df[["GUESTS"]]

    X, y = normalize_features_and_targets(X_df, y_df, is_train)

    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

    return TensorDataset(X, y)

def articles_df_to_tensor_dataset(df: pd.DataFrame, is_train: bool = True) -> TensorDataset:
    """
    Converts a DataFrame into a PyTorch TensorDataset for article sales prediction.

    Assumes:
    - All columns are numeric.
    - Article columns start with "art_".
    - Article columns are the targets.

    Returns:
        TensorDataset: Features (X) and article targets (y).
    """
    article_cols = [col for col in df.columns if col.startswith("art_")]
    X_df = df.drop(columns=article_cols)
    y_df = df[article_cols]

    X, y = normalize_features_and_targets(X_df, y_df, is_train)

    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32)
    return TensorDataset(X, y)

def guest_and_articles_df_to_tensor_dataset(df: pd.DataFrame, is_train: bool = True) -> TensorDataset:
    """
    Converts a DataFrame into a PyTorch TensorDataset for joint guest and article prediction.

    Assumes:
    - All columns are numeric.
    - 'GUESTS' and columns starting with 'art_' are targets.
    - All other columns are input features.

    Returns:
        TensorDataset: Features (X) and multi-task targets (y).
    """
    target_cols = ['GUESTS'] + [col for col in df.columns if col.startswith("art_")]
    X_df = df.drop(columns=target_cols)
    y_df = df[target_cols]

    X, y = normalize_features_and_targets(X_df, y_df, is_train)

    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32)
    return TensorDataset(X, y)

def guest_and_articles_df_to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    """
    Converts a DataFrame into a PyTorch TensorDataset for joint guest and article prediction.

    Assumes:
    - All columns are numeric.
    - 'GUESTS' and columns starting with 'art_' are targets.
    - All other columns are input features.

    Returns:
        TensorDataset: Features (X) and multi-task targets (y).
    """
    target_cols = ['GUESTS'] + [col for col in df.columns if col.startswith("art_")]
    X = torch.tensor(df.drop(columns=target_cols).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df[target_cols].to_numpy(), dtype=torch.float32)
    return TensorDataset(X, y)


def prepare_dataloader(df: pd.DataFrame,
                       batch_size: int = 64,
                       to_tensor_fn: Callable[[pd.DataFrame], TensorDataset] 
                       = guest_df_to_tensor_dataset,
                       is_train: bool = True
                       ) -> DataLoader:
    """
    Prepares a PyTorch DataLoader for training or evaluation.

    This function performs the following steps:
    - Extracts additional features from the 'Date' column (year, month, day_of_year)
    - Normalizes continuous numeric features
    - Converts the processed DataFrame into a TensorDataset using a provided conversion function
    - Wraps the dataset in a DataLoader for model consumption

    Args:
        df (pd.DataFrame): The input DataFrame containing features and targets.
        batch_size (int): Number of samples per batch for the DataLoader.
        to_tensor_fn (Callable[[pd.DataFrame], TensorDataset]): 
            A function that takes a DataFrame and returns a TensorDataset. 
            This allows flexibility for different tasks. For example:
            - Use `guest_df_to_tensor_dataset` for guest prediction tasks.
            - Use `articles_df_to_tensor_dataset` for article sales prediction tasks.
            - Use `guest_and_articles_df_to_tensor_dataset` for joint guest and article prediction tasks.
            Ensure the function matches the structure of the DataFrame provided.

    Returns:
        DataLoader: A PyTorch DataLoader ready for training or validation.
    """
    df = split_date(df, drop_date=True)
    ds = to_tensor_fn(df, is_train=is_train)

    # Wrap in DataLoaders (shuffle if training)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train)

    return loader