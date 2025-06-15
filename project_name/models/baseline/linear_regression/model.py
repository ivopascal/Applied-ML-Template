from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple
import time
import numpy as np
import joblib


class LinearModelHandler:
    """model handeler for linear regression"""
    def __init__(self) -> None:
        """Init handler
        """
        self.model: LinearRegression = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """train model

        Args:
            X_train (np.ndarray): input
            y_train (np.ndarray): ground truth
        """
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """predict

        Args:
            X_test (np.ndarray): observation

        Returns:
            np.ndarray: prediction
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_test)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """eval model

        Args:
            X_test (np.ndarray): observation
            y_test (np.ndarray): ground truth

        Returns:
            Tuple[float, float, float, float, float, float, float, float]:
            results
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        start_time = time.time()
        y_pred = self.model.predict(X_test)
        end_time = time.time()

        mse_score = mean_squared_error(y_test, y_pred)
        rmse_score = np.sqrt(mse_score)
        mae_score = mean_absolute_error(y_test, y_pred)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        abs_rel = np.mean(np.abs(y_pred - y_test) / (np.abs(y_test) + epsilon))

        # Threshold Accuracy Metrics
        denom1 = y_test + epsilon
        denom2 = y_pred + epsilon
        ratios = np.maximum(y_pred / denom1, y_test / denom2)

        delta1 = np.mean(ratios < 1.25)
        delta2 = np.mean(ratios < 1.25 ** 2)
        delta3 = np.mean(ratios < 1.25 ** 3)

        inference_time = end_time - start_time

        return (mse_score, rmse_score, mae_score,
                abs_rel, delta1, delta2, delta3, inference_time
                )

    def save_model(self, path: str = "trained_linear_model.pkl") -> None:
        """save model

        Args:
            path (str, optional): path to save.
            Defaults to "trained_linear_model.pkl".
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "trained_linear_model.pkl") -> None:
        """load model

        Args:
            path (str, optional): path to load model.
            Defaults to "trained_linear_model.pkl".
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
