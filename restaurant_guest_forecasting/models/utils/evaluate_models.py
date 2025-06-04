import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.models.random_guesser.\
    random_regression_guesser import RandomRegressionGuesser
from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data
from restaurant_guest_forecasting.data.normalization import normalize_df, preprocess_df
from restaurant_guest_forecasting.data.normalizer.normalizer import Normalizer


def test_mse(model: RandomRegressionGuesser | LinearRegression):
    _, _, test_data = train_val_test_data()
    X, y = preprocess_df(test_data)
    predictions = model.predict(X)
        # normalizer = Normalizer(is_target=True)
        # normalizer.load()
        # predictions = normalizer.inverse_transform(y)

    

    return mean_squared_error(y, predictions)
