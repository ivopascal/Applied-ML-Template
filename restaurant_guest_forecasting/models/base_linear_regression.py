import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from train_test_split import train_val_test_data
from  normalization import normalize_data
from split_date import split_date

def train(X_train, y_train, path: None, save_model: bool = True):
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Saving the model
    if save_model:
        pass

def load_model(path):
    # Open the model

    # Return the model
    pass

def evaluate(model, X_test, y_test):
    # Predict based on the test data
    y_pred = model.predict(X_test)

    ## Evaluation
    # Mean squared error to obtain the accuracy
    m_squared_error = mean_squared_error(y_test, y_pred)
    print(f"Base linear regression model accuracy: {m_squared_error}")
    
    # Coefficients correspond to feature importance
    importance = model.coef_

    # Assuming X_train is a DataFrame, you can pair feature names with their 
    # coefficients
    feature_importance = dict(zip(X_train.columns, importance))

    # Print features, sorted based on importance
    print("---feature importance---")
    for feature, coef in sorted(feature_importance.items(), key=lambda 
                                x: abs(x[1]), reverse=True):
        print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    full_restaurant_df = train_val_test_data()
    regression_df = []

    # Dropping article columns for train, val and test dataframes
    for dataframe in full_restaurant_df:
        dropped_cols_df = dataframe.drop(columns=[col for col in 
                                                  dataframe.columns if 
                                                  col.startswith('art_')])
        regression_df.append(dropped_cols_df)

    train_data, val, test = regression_df

    # Combine test and validation sets, as we don't need the validation set
    test_data = pd.concat([val, test], ignore_index=True)
    test_data = test_data.sort_values(by='Date')

    train_data = split_date(train_data, drop_date = True)
    test_data = split_date(test_data, drop_date = True)

    # Normalization
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    # Split into X and y
    X_train, y_train = train.drop(columns="GUESTS"), train_data["GUESTS"]
    X_test, y_test = test_data.drop(columns="GUESTS"), test_data["GUESTS"]

    train(X_train, y_train)
