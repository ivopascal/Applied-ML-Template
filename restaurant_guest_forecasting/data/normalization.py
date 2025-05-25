from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize continuous numeric features in the input DataFrame using Min-Max scaling.

    This function applies MinMaxScaler to scale specified continuous features to the range [0, 1],
    while leaving all other columns (e.g., categorical, one-hot encoded, or identifiers) unchanged.

    Args:
        data (pd.DataFrame): The input DataFrame containing all features.

    Returns:
        pd.DataFrame: A new DataFrame where the specified continuous features are scaled,
                      and all other columns are preserved as-is.
    """
    # List of known continuous features to scale
    continuous_columns = [
        'GUESTS',
        'tempmax',
        'tempmin',
        'temp',
        'feelslikemax',
        'feelslikemin',
        'feelslike',
        'humidity',
        'precip',
        'precipprob',
        'windgust',
        'windspeed',
        'cloudcover',
        'solarradiation',
        'uvindex'
    ]

    # Create a copy to avoid modifying the original DataFrame
    scaled_df = data.copy()

    # Initialize MinMaxScaler and apply it only to the continuous columns
    scaler = MinMaxScaler()
    scaled_df[continuous_columns] = scaler.fit_transform(data[continuous_columns])

    return scaled_df

if __name__ == '__main__':
    restaurant_df = pd.read_csv('restaurant_guest_forecasting/data/cleaned/full_restaurant_data_rank_int.csv')
    scaled_restaurant_df = normalize_data(restaurant_df)
