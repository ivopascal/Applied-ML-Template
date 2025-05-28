import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import pickle
from sklearn.metrics import mean_squared_error

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.normalization import normalize_data
from restaurant_guest_forecasting.data.split_date import split_date

def train_linear_regression():
    print("Training LinearRegression...")

    # Path to save the model
    model_path = os.path.join(
        os.path.dirname(__file__), "saved_models", "linear_regression_model.pkl"
    )

    # Load and split data
    train_data, val_data, _ = train_val_test_data()
    train_data = split_date(train_data, drop_date=True)
    val_data = split_date(val_data, drop_date=True)

    X_train, y_train = train_data.drop(columns=['GUESTS']), train_data['GUESTS']
    X_val, y_val = val_data.drop(columns=['GUESTS']), val_data['GUESTS']

    # Train and evaluate
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    print(f"LinearRegression saved to {model_path}")
    print(f"Train MSE: {mean_squared_error(y_train, train_preds):.2f}")
    print(f"Validation MSE: {mean_squared_error(y_val, val_preds):.2f}")


train_linear_regression()
