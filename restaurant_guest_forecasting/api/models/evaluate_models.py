import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.models.random_guesser.\
    random_regression_guesser import RandomRegressionGuesser
from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data, print_df_details
from restaurant_guest_forecasting.data.split_date import split_date

def validation_mse(model: RandomRegressionGuesser | LinearRegression):
    _, val_data, test_data = train_val_test_data()
    val_test_data = pd.concat([val_data, test_data], axis=0).sort_values("Date").reset_index(drop=True)
    print(val_data.head())
    val_data = split_date(val_data)

    print(val_data.head())

    X, y = val_data.drop(columns=['GUESTS']), val_data['GUESTS']

    predictions = model.predict(X)
    return mean_squared_error(y, predictions)
