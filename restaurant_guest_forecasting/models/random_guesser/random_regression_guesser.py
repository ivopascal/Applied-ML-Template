import pandas as pd

class RandomRegressionGuesser:
    def __init__(self) -> None:
        self._predicted_value = None

    def train(self, X_unused: pd.DataFrame, y: pd.DataFrame) -> None:
        self._predicted_value = int(y.mean())

    def predict(self, X_unused: pd.DataFrame) -> int:
        if self._predicted_value is None:
            raise RuntimeError("The model has not been trained, train it first, then predict.")
        return self._predicted_value
    import pandas as pd

class RandomRegressionGuesser:
    """A simple regression model that always predicts the mean of the training targets."""

    def __init__(self) -> None:
        """Initializes the RandomRegressionGuesser.

        Sets the predicted value to None until training is done.
        """
        self._predicted_value = None

    def train(self, X_unused: pd.DataFrame, y: pd.DataFrame) -> None:
        """Trains the model by computing the mean of the target variable.

        This model ignores the features and always predicts the integer mean of the target.

        Args:
            X_unused (pd.DataFrame): Input features (not used in training).
            y (pd.DataFrame): Target values used to compute the prediction value.
        """
        self._predicted_value = int(y.mean())

    def predict(self, X_unused: pd.DataFrame) -> int:
        """Predicts the same value (mean of training targets) for all inputs.

        Args:
            X_unused (pd.DataFrame): Input features (not used in prediction).

        Returns:
            int: The predicted value (mean of training targets).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._predicted_value is None:
            raise RuntimeError("The model has not been trained, train it first, then predict.")
        return self._predicted_value
