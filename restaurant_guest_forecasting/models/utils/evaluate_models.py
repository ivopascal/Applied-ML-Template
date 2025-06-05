import torch
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.models.random_guesser.\
    random_regression_guesser import RandomRegressionGuesser
from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data
from restaurant_guest_forecasting.data.normalization import normalize_df, preprocess_df
from restaurant_guest_forecasting.data.normalizer.normalizer import Normalizer

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader


def test_mse(model: RandomRegressionGuesser | LinearRegression):
    _, _, test_data = train_val_test_data()
    X, y = preprocess_df(test_data)
    predictions = model.predict(X)

    

    return mean_squared_error(y, predictions)

def test_mlp_mse(model: MultiTaskMLP):
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
            output = normalizer.inverse_transform_value(output.item())
            predictions.append(output)
    
    return mean_squared_error(y_df, predictions)
