import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.split_date import split_date
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader,\
                                                    guest_df_to_tensor_dataset

from restaurant_guest_forecasting.data.normalization import preprocess_df

from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser\
      import RandomRegressionGuesser


def train_and_save_model(model: RandomRegressionGuesser | LinearRegression,
                         model_filename: str,
                         test: bool = True) -> None:
    """Train and save a given model with optional evaluation.

    Args:
        model: A model object with `.fit()` and `.predict()` methods.
        model_filename: Name of the file to save the trained model to.
        evaluate: Whether to print training and validation MSE.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_models", model_filename)

    # Load and split data
    train_data, _, test_data = train_val_test_data()
    X_train, y_train = preprocess_df(train_data)
    X_test, y_test = preprocess_df(test_data)

    
    # Train model
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

    if test:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        print(f"Train MSE: {mean_squared_error(y_train, train_preds):.2f}")
        print(f"Test MSE: {mean_squared_error(y_test, test_preds):.2f}")


