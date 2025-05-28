import pandas as pd
import numpy as np
from typing import List

from sklearn.metrics import mean_squared_error

class RandomRegressionGuesser:
    """A simple regression model that always predicts the mean of the training targets."""

    def __init__(self) -> None:
        """Initializes the RandomRegressionGuesser.

        Sets the predicted value and mse to None until training is done.
        """
        self._predicted_value = None
        self._train_mse = None

    def train(self, X_unused: pd.DataFrame, y: pd.DataFrame) -> None:
        """Trains the model by computing the mean of the target variable.

        This model ignores the features and always predicts the integer mean of the target.

        Args:
            X_unused (pd.DataFrame): Input features (not used in training).
            y (pd.DataFrame): Target values used to compute the prediction value.
        """

        self._predicted_value = int(y.mean())

        predictions = [self._predicted_value] * len(X_unused)

        self._train_mse = mean_squared_error(y.to_numpy().flatten(),
                                              predictions)

    def predict(self, X: pd.DataFrame) -> List[int]:
        """Predicts the same value (mean of training targets) for all inputs.

        Args:
            X (pd.DataFrame): Input features (not used in prediction).

        Returns:
            int: The predicted value (mean of training targets).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._predicted_value is None:
            raise RuntimeError("The model has not been trained, train it first, then predict.")
        return [self._predicted_value] * len(X)
    

    @property
    def train_mse(self) -> float:
        """Returns the mean squared error from training.

        Returns:
            float: Mean squared error.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._train_mse is None:
            raise RuntimeError("The model has not been trained, train it first.")
        return self._train_mse
