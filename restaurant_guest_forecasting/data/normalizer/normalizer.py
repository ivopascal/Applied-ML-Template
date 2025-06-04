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

    TARGET_COLUMNS_SINGLE_TASK = ['GUESTS']

    TRAIN_PATH = "restaurant_guest_forecasting/data/normalizer/scaler_train.pkl"
    TARGET_PATH = "restaurant_guest_forecasting/data/normalizer/scaler_target.pkl"


    def __init__(self, is_target: bool = False):
        """
        Initializes the Normalizer with a list of continuous columns to scale.

        Args:
            continuous_columns (list): List of column names to be normalized.
        """
        self.is_target = is_target
        self.continuous_columns = Normalizer.TARGET_COLUMNS_SINGLE_TASK if is_target\
              else Normalizer.CONTINUOUS_COLUMNS
        self.scaler = MinMaxScaler()

    def save(self):
        """
        Saves the internal scaler to a file.
        """
        path = Normalizer.TARGET_PATH if self.is_target else Normalizer.TRAIN_PATH
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self):
        """
        Loads the scaler from a file. Raises FileNotFoundError if the file does not exist.
        """
        path = Normalizer.TARGET_PATH if self.is_target else Normalizer.TRAIN_PATH

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

        missing = [col for col in self.continuous_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in input data for normalization: {missing}")

        scaled_data = data.copy()
        if isinstance(scaled_data[self.continuous_columns], pd.Series):
            # If only one continuous column, convert to DataFrame
            scaled_data[self.continuous_columns] = scaled_data[self.continuous_columns].to_frame()  
        scaled_data[self.continuous_columns] = self.scaler.transform(data[self.continuous_columns])
        return scaled_data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms the DataFrame by reverting the scaling of the specified continuous columns.

        Args:
            data (pd.DataFrame): The input DataFrame containing all features.

        Returns:
            pd.DataFrame: A new DataFrame with inverse transformed continuous features.
        """
        scaled_data = data.copy()
        if isinstance(scaled_data[self.continuous_columns], pd.Series):
            # If only one continuous column, convert to DataFrame
            scaled_data[self.continuous_columns] = scaled_data[self.continuous_columns].to_frame()  
        scaled_data[self.continuous_columns] = self.scaler.inverse_transform(data[self.continuous_columns])
        return scaled_data