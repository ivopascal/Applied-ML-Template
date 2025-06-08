import pandas as pd

def split_date(X_orig: pd.DataFrame, drop_date: bool = False) -> pd.DataFrame:
    """
    Splits the "Date" column from features in "year", "month" and "day_of_year".
    Optionally also drops the original "Date" column.

    Args:
        X_orig (pd.DataFrame): Original features DataFrame.
        drop_date (bool): Drop the "Date" column from the returned DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame with columns added (and optionally "Date" column removed).

    Raises:
        ValueError: If essential columns like 'Date' are missing or if other
                    defined features are not found in X_raw after initial processing.
    """
    X_processed = X_orig.copy()

    # Convert "Date" column to datetime objects and extract features
    if "Date" not in X_processed.columns:
        raise ValueError(
            "Missing 'Date' column in X_orig. Cannot derive date features."
        )

    try:
        X_processed["Date"] = pd.to_datetime(X_processed["Date"])
    except Exception as e:
        raise ValueError(
            f"Error converting 'Date' column to datetime: {e}. Ensure it's in ISO format."
        )

    X_processed["year"] = X_processed["Date"].dt.year
    X_processed["month"] = X_processed["Date"].dt.month
    X_processed["day_of_year"] = X_processed["Date"].dt.dayofyear

    # Drop the original 'Date' column as its information is now in derived features
    if drop_date and "Date" in X_processed.columns:
        X_processed = X_processed.drop("Date", axis=1)

    X_processed = X_processed.reindex(sorted(X_processed.columns), axis=1)

    return X_processed
