import os
import pickle
import sys

# # Add project root to PYTHONPATH manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser import RandomRegressionGuesser

def train_random_regression_guesser() -> None:
    """Train and save the RandomRegressionGuesser model."""
    # Define model-specific path
    model_path = os.path.join(
        os.path.dirname(__file__), "saved_models", "random_regression_guesser.pkl"
    )

    # Prepare training data
    train_data, _, _ = train_val_test_data()
    train_data = split_date(train_data)
    X, y = train_data.drop(columns=['GUESTS']), train_data['GUESTS']

    # Train model
    rg = RandomRegressionGuesser()
    rg.train(X, y)

    # Ensure directory exists and save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(rg, f)

    print(f"RandomRegressionGuesser model saved to {model_path}")

def main():
    train_random_regression_guesser()

if __name__ == "__main__":
    main()