import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.split_date import split_date

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP

import numpy as np
import pandas as pd

import shap

import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

class ExplainPredictions():
    def __init__(self, 
                 model: nn.Module,
                 train_data: DataLoader,
                 test_data: DataLoader):
        
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def predict_wrapper(self, X, device='cpu'):
        self.model.to(device)
        self.model.eval()

        predictions = []

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = self.model(X_tensor)
            if isinstance(outputs, list):
                outputs = outputs[0] 
            predictions = outputs.detach().cpu().numpy().flatten()
        
        return predictions

    def _calculate_shap_values(self):
        kernel_explainer = shap.KernelExplainer(self.predict_wrapper, self.train_data.iloc[:100, :])
        shap_values = kernel_explainer(self.test_data.iloc[51:52, :])
        
        return shap_values

    def shap_plot(self):
        shap_values = self._calculate_shap_values()

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0] * 1000, show=False)
        plt.title("PartitionExplainer for instance 0 with the KernelExplainer")
        plt.show()

if __name__ == "__main__":
    train_df, _, test_df = train_val_test_data()

    train_df = split_date(train_df, drop_date=True)
    test_df = split_date(test_df, drop_date=True)

    art_columns=[col for col in train_df.columns if col.startswith("art_")]
    train_df, test_df = train_df.drop(columns=art_columns), \
                        test_df.drop(columns=art_columns), 

    X_train, y_train = train_df.drop(columns=['GUESTS']), train_df['GUESTS']
    X_test, y_test = test_df.drop(columns=['GUESTS']), test_df['GUESTS']

    # Order the columns, so the order always matches
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)

    normalizer = Normalizer()
    X_train = pd.DataFrame(
        normalizer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        normalizer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    input_size = 37
    neurons = [input_size, input_size, input_size]

    single_task_mlp = MultiTaskMLP(num_neurons=neurons, 
                                   droput_rate=0.0, 
                                   activation="relu", 
                                   output_neurons=[1])

    predictor = ExplainPredictions(single_task_mlp, X_train, X_test)
    # predictions = predictor.predict_wrapper(X_test)
    predictor.shap_plot()
