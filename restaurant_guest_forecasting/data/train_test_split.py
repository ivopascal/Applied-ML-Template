import pandas as pd
import os
from typing import Tuple

def train_val_test_data(rank_enc: str = "int", regression: bool = True) \
                          -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits restaurant data into training, validation, and test sets.

    The function loads preprocessed restaurant data with either integer or 
    one-hot rank encodings, and splits the data as follows:
    - The last 365 rows are reserved for validation and test sets.
    - Validation receives even-indexed rows, and test receives odd-indexed rows.
    - The remaining rows are used for training.

    Args:
        rank_enc (str): Type of rank encoding to use. Must be either:
            - "int": Integer-encoded articles sold.
            - "int_rank": Alternate CSV file with integer-encoded ranks.
            - "onehot_rank": One-hot encoded ranks (loaded from Pickle).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - train_df (DataFrame): Training set (all but last 365 rows).
            - val_df (DataFrame): Validation set (even-indexed rows from last 365).
            - test_df (DataFrame): Test set (odd-indexed rows from last 365).

    Raises:
        ValueError: If `rank_enc` is not one of "int" or "onehot".
    """
    size_val_test = 365  # Number of rows taken for val/test
    # Get the path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "cleaned")

    if rank_enc == "int":
        path = os.path.join(base_dir, "full_restaurant_data_sold_articles_int.csv")
        data = pd.read_csv(path)
    elif rank_enc == "int_rank":
        path = os.path.join(base_dir, "full_restaurant_data_rank_int.csv")
        data = pd.read_csv(path)
    elif rank_enc == "onehot_rank":
        path = os.path.join(base_dir, "full_restaurant_data_rank_1hot.pkl")
        data = pd.read_pickle(path)
    else:
        raise ValueError("`rank_enc` must be 'int', 'int_rank' or 'onehot_rank'")

    num_rows = data.shape[0]

    if regression:
        data = data.drop(columns=[col for col in data.columns if col.startswith("art_")], errors='ignore')
    else:
        data = data.drop(columns=["GUESTS"], errors='ignore')
        

    train_df = data.head(num_rows - size_val_test).reset_index(drop=True)
    val_test_data = data.tail(size_val_test).reset_index(drop=True)

    val_df = val_test_data.iloc[::2].reset_index(drop=True)  # Even indices
    test_df = val_test_data.iloc[1::2].reset_index(drop=True)  # Odd indices

    return train_df, val_df, test_df


def print_df_details(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Prints basic information about a DataFrame."""
    print(f"\n{name}:")
    print(f"Shape: {df.shape}")
    print("Head:")
    print(df.head(3))
    print("Tail:")
    print(df.tail(3))


def main():
    # Load splits
    split_int = train_val_test_data("int")
    split_int_rank = train_val_test_data("int_rank")
    split_onehot_rank = train_val_test_data("onehot_rank")

    # Print details for integer-encoded data
    print("== Integer-Encoded Data ==")
    for name, df in zip(["Train", "Validation", "Test"], split_int):
        print_df_details(df, name)

    # Print details for integer-encoded ranked data
    # print("== Integer-Encoded Rank Data ==")
    # for name, df in zip(["Train", "Validation", "Test"], split_int_rank):
    #     print_df_details(df, name)


    # Optionally, also inspect the one-hot encoded split
    # print("== One-Hot Encoded Rank Data ==")
    # for name, df in zip(["Train", "Validation", "Test"], split_onehot_rank):
    #     print_df_details(df, name)


if __name__ == "__main__":
    main()