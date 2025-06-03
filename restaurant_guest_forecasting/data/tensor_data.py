import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from typing import Callable
from restaurant_guest_forecasting.data.normalization import normalize_data
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.data.normalizer.normalizer import Normalizer



def guest_df_to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    """
    Converts a DataFrame into a PyTorch TensorDataset for guest prediction. 
    Also drops the article columns "art_<article>"

    Assumes:
    - All columns are numeric
    - 'GUESTS' is the target column
    """
    art_columns=[col for col in df.columns if col.startswith("art_")]
    df = df.drop(columns=art_columns)

    X = torch.tensor(df.drop(columns=["GUESTS"]).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["GUESTS"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    return TensorDataset(X, y)

def articles_df_to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
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
    X = torch.tensor(df.drop(columns=article_cols).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df[article_cols].to_numpy(), dtype=torch.float32)
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
        shuffle (bool): Whether to shuffle the data in the DataLoader.
        to_tensor_fn (Callable[[pd.DataFrame], TensorDataset]): 
            A function that takes a DataFrame and returns a TensorDataset. 
            This allows flexibility for different tasks.

    Returns:
        DataLoader: A PyTorch DataLoader ready for training or validation.
    """
    df = split_date(df, drop_date=True)

    normalizer = Normalizer()
    if is_train:
        normalizer.fit(df, save=True)
    else:
        normalizer.load()
    df = normalizer.transform(df)

    ds = to_tensor_fn(df)

    # Wrap in DataLoaders (shuffle if training)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train)

    return loader