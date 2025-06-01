import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

from restaurant_guest_forecasting.models.losses.asymmetric_loss \
    import AsymmetricL2MSE

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP, MLPBase

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader,\
                                                    guest_df_to_tensor_dataset

from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser\
      import RandomRegressionGuesser

def train_and_save_model(model: RandomRegressionGuesser | LinearRegression,
                         model_filename: str,
                         evaluate: bool = True) -> None:
    """Train and save a given model with optional evaluation.

    Args:
        model: A model object with `.fit()` and `.predict()` methods.
        model_filename: Name of the file to save the trained model to.
        evaluate: Whether to print training and validation MSE.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_models", model_filename)

    # Load and split data
    train_data, val_data, _ = train_val_test_data()
    train_data = split_date(train_data, drop_date=True)
    val_data = split_date(val_data, drop_date=True)

    X_train, y_train = train_data.drop(columns=['GUESTS']), train_data['GUESTS']
    X_val, y_val = val_data.drop(columns=['GUESTS']), val_data['GUESTS']

    # Order the columns, so the order always matches
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_val = X_val.reindex(sorted(X_val.columns), axis=1)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

    if evaluate:
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        print(f"Train MSE: {mean_squared_error(y_train, train_preds):.2f}")
        print(f"Validation MSE: {mean_squared_error(y_val, val_preds):.2f}")


# def train_save_mlp(model: MultiTaskMLP)


def main():
    train_and_save_model(RandomRegressionGuesser(),
                         "random_regression_guesser.pkl")

    train_and_save_model(LinearRegression(),
                         "linear_regression.pkl")


if __name__ == "__main__":
    main()