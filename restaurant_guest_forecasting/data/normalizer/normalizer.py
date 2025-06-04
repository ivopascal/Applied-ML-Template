import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

import os

class Normalizer:

    CONTINUOUS_COLUMNS = [
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

    def __init__(self, continuous_columns: list = CONTINUOUS_COLUMNS):
        """
        Initializes the Normalizer with a list of continuous columns to scale.

        Args:
            continuous_columns (list): List of column names to be normalized.
        """
        self.continuous_columns = continuous_columns
        self.scaler = MinMaxScaler()

    def save(self, path: str = "restaurant_guest_forecasting/data/normalizer/scaler.pkl"):
        """
        Saves the internal scaler to a file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, path: str = "restaurant_guest_forecasting/data/normalizer/scaler.pkl"):
        """
        Loads the scaler from a file. Raises FileNotFoundError if the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at '{path}'. Make sure to fit and save it first.")
        
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)

    def fit(self, data: pd.DataFrame, save: bool = True) -> None:
        """
        Fits the MinMaxScaler to the specified continuous columns in the DataFrame.
        Saves the fitted scaler to a file for later use.
        This method should be called once with the entire dataset to learn the scaling parameters.

        Args:
            data (pd.DataFrame): The input DataFrame containing all features.
            save (bool): Whether to save the fitted scaler to a file after fitting.
        """
        self.scaler.fit(data[self.continuous_columns])
        if save:
            self.save()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by scaling the specified continuous columns.

        Args:
            data (pd.DataFrame): The input DataFrame containing all features.

        Returns:
            pd.DataFrame: A new DataFrame with scaled continuous features.
        """
        scaled_data = data.copy()
        scaled_data[self.continuous_columns] = self.scaler.transform(data[self.continuous_columns])
        return scaled_data