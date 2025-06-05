import pickle
import os

from typing import Union, List, Literal
import torch

from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP

from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser import RandomRegressionGuesser

def load_model(model_file_name: str) -> Union[RandomRegressionGuesser,
                                              LinearRegression]:
    """Load the saved RandomRegressionGuesser model."""
    model_path = os.path.join(
        os.path.dirname(__file__), "saved_models", model_file_name
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_mlp(num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu",
                 output_neurons: List[int] = [1]) -> MultiTaskMLP:
    """    Load a MultiTaskMLP model with the specified architecture."""

    model = MultiTaskMLP(num_neurons, droput_rate, activation, output_neurons)
    model_path = os.path.join(
        os.path.dirname(__file__), "saved_models", "guests_mlp.pt"
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    return model

