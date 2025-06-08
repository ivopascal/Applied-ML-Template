import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.models.random_guesser.\
    random_regression_guesser import RandomRegressionGuesser
from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data
from restaurant_guest_forecasting.data.normalization import normalize_df, preprocess_df
from restaurant_guest_forecasting.data.normalizer.normalizer import Normalizer

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.models.losses.asymmetric_loss import AsymmetricL2MSE
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader

DTYPE = torch.float64


def test_mse(model: RandomRegressionGuesser | LinearRegression):
    _, _, test_data = train_val_test_data()
    X, y = preprocess_df(test_data)
    predictions = [int(value) for value in model.predict(X)]

    return mean_squared_error(y, predictions)

def test_mlp_mse(model: MultiTaskMLP, normalized: bool = True):
    """
    Tests the MSE of a MultiTaskMLP model on the test dataset.
    
    Args:
        model (MultiTaskMLP): The trained MultiTaskMLP model.
    
    Returns:
        float: The mean squared error of the model's predictions on the test set.
    """
    _, _, test_data = train_val_test_data()
    _, y_df = preprocess_df(test_data)

    loader = prepare_dataloader(test_data, batch_size=1, is_train=False)

    if normalized:
        normalizer = Normalizer(is_target=True)
        normalizer.load()

    with torch.no_grad():
        model.eval()
        predictions = []
        for X, _ in loader:
            output = model(X)
            if isinstance(output, list):
                # If the model returns a list, take the first element
                output = output[0]
            output = normalizer.inverse_transform_value(output.item()) if normalized else output.item()
            predictions.append(output)
    
    return mean_squared_error(y_df.to_numpy(), np.array(predictions))

def test_mlp_asymmetric_mse(model: MultiTaskMLP, normalized: bool = True):
    """
    Tests the AsymmetricL2MSE of a MultiTaskMLP model on the test dataset.

    Args:
        model (MultiTaskMLP): The trained MultiTaskMLP model.

    Returns:
        float: The AsymmetricL2MSE of the model's predictions on the test set.
    """
    _, _, test_data = train_val_test_data()
    _, y_df = preprocess_df(test_data)

    loader = prepare_dataloader(test_data, batch_size=1, is_train=False)

    if normalized:
        normalizer = Normalizer(is_target=True)
        normalizer.load()

    criterion = AsymmetricL2MSE(l2_lambda=0, w_over=2, w_under=1)

    predictions = []
    with torch.no_grad():
        model.eval()
        for X, _ in loader:
            output = model(X)
            if isinstance(output, list):
                output = output[0]
            output = normalizer.inverse_transform_value(output.item()) if normalized else output.item()
            predictions.append(output)

    y_true = torch.tensor(y_df.to_numpy(), dtype=DTYPE, requires_grad=False)
    y_pred = torch.tensor(np.array(predictions), dtype=DTYPE, requires_grad=False)
    loss = criterion(y_pred, y_true)
    return loss.item()

def test_asymmetric_mse(model: RandomRegressionGuesser | LinearRegression):
    """
    Tests the AsymmetricL2MSE of a RandomRegressionGuesser or LinearRegression model on the test dataset.

    Args:
        model (RandomRegressionGuesser | LinearRegression): The trained model.

    Returns:
        float: The AsymmetricL2MSE of the model's predictions on the test set.
    """
    _, _, test_data = train_val_test_data()
    X, y = preprocess_df(test_data)

    predictions = [int(value) for value in model.predict(X)]

    y_true = torch.tensor(y.to_numpy(), dtype=DTYPE, requires_grad=False)
    y_pred = torch.tensor(predictions, dtype=DTYPE, requires_grad=False)

    criterion = AsymmetricL2MSE(l2_lambda=0, w_over=2, w_under=1)
    loss = criterion(y_pred, y_true)
    return loss.item()