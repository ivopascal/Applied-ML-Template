import pickle
import os

from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser import RandomRegressionGuesser

def load_random_regression_guesser() -> RandomRegressionGuesser:
    """Load the saved RandomRegressionGuesser model."""
    model_path = os.path.join(
        os.path.dirname(__file__), "saved_models", "random_regression_guesser.pkl"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
