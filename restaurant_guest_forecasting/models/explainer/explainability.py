import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

import torch
from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP

from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data
from restaurant_guest_forecasting.data.normalization import \
    normalize_df
import os


class PredictionsExplainer():
    def __init__(self,
                 model: MultiTaskMLP,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame) -> None:
        """
        Initialize the PredictionsExplainer.

        Args:
            model (MultiTaskMLP): The trained model to explain.
            train_data (Union[pd.DataFrame, np.ndarray]): Training data for
            SHAP background.
            test_data (Union[pd.DataFrame, np.ndarray]): Test data for
            generating SHAP values.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        if not isinstance(train_data, (pd.DataFrame)):
            raise TypeError("train_data must be a pandas DataFrame")

        if not isinstance(test_data, (pd.DataFrame, np.ndarray)):
            raise TypeError("test_data must be a pandas DataFrame or a \
                            NumPy array")

        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def _predict_wrapper(self,
                         X: np.ndarray,
                         device='cpu') -> np.ndarray:
        """
        Wraps the model prediction for SHAP KernelExplainer.

        Args:
            X (np.ndarray): Input data.
            device (str): Device to use for computation ('cpu' or 'cuda').

        Returns:
            np.ndarray: Model predictions.
        """
        self.model.to(device)
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for row in range(X.shape[0]):
                X_tensor = torch.tensor(X[row], dtype=torch.float32).\
                    unsqueeze(0).to(device)
                outputs = self.model(X_tensor)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                pred = outputs.detach().cpu().numpy().flatten()
                predictions.append(pred)

        return np.array(predictions).flatten()

    def _calculate_shap_values(self,
                               sample_size: int = 50,
                               test_index: int = 100) -> shap.Explanation:
        """
        Calculate SHAP values using KernelExplainer.

        Args:
            sample_size (int): Number of samples to use for background data.
            test_index (int): Index of test sample to explain.

        Returns:
            shap.Explanation: SHAP values for the selected test instance.
        """
        if sample_size > len(self.train_data):
            raise ValueError("sample_size exceeds the number of available \
                             training samples.")

        if test_index >= len(self.test_data):
            raise IndexError("test_index is out of bounds for the test data.")

        background_data = self.train_data.sample(n=sample_size,
                                                 random_state=42)
        test_instance = self.test_data.iloc[test_index:test_index + 1, :]\
            .to_numpy()

        explainer = shap.KernelExplainer(self._predict_wrapper,
                                         background_data)
        shap_values = explainer(test_instance)

        return shap_values

    def shap_plot(self,
                  sample_size: int = 50,
                  test_index: int = 100,
                  scale_factor: float = 1000.0) -> None:
        """
        Generate and display a SHAP waterfall plot.

        Args:
            sample_size (int): Number of samples for SHAP background.
            test_index (int): Index of test instance to explain.
            scale_factor (float): Factor to scale SHAP values for better
            visualization.
        """
        shap_values = self._calculate_shap_values(sample_size=sample_size,
                                                  test_index=test_index)

        shap.plots.waterfall(shap_values[0] * scale_factor, show=False)
        plt.title(
            f"KernelExplainer SHAP Waterfall Plot \
            (background size: {sample_size}, test index: {test_index})"
        )
        plt.show()


if __name__ == "__main__":
    # Loading train and test datasets
    train_df, _, test_df = train_val_test_data()

    # Data normalization
    X_train, _ = normalize_df(train_df, is_train=True)
    X_test, _ = normalize_df(test_df, is_train=False)

    # Initializing the model with the same architecture as the trained model
    input_size = X_train.shape[1]
    neurons = [input_size] + [1024]*6 + [512, 256, 128]

    single_task_mlp = MultiTaskMLP(num_neurons=neurons,
                                   droput_rate=0.0,
                                   activation="relu",
                                   output_neurons=[1])

    # Setting the path of the saved model
    model_path = os.path.join(os.path.dirname(__file__),
                              "..",
                              "utils",
                              "saved_models",
                              "guests_mlp.pt")

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cpu')

    # Load into model
    single_task_mlp.load_state_dict(state_dict)

    # Making a shap plot for a single prediction
    predictor = PredictionsExplainer(single_task_mlp, X_train, X_test)
    predictor.shap_plot(sample_size=10)
