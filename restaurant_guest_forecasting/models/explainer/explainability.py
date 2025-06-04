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

class PredictionsExplainer():
    def __init__(self, 
                 model: MultiTaskMLP,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader) -> None:
        
        self.model = model
        self.train_data = train_data_loader
        self.test_data = test_data_loader





    def _predict_wrapper(self, X: np.ndarray, device='cpu') -> np.ndarray:
        self.model.to(device)
        self.model.eval()
    
        predictions = []

        with torch.no_grad():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
            print(X_tensor[0], X_tensor[0].shape)
            outputs = self.model(X_tensor[0])
            if isinstance(outputs, list):
                outputs = outputs[0] 
            predictions = outputs.detach().cpu().numpy().flatten()
        
        return predictions

    def _calculate_shap_values(self):
        sampled_data = self.train_data.sample(n=100, random_state=42)
        kernel_explainer = shap.KernelExplainer(self._predict_wrapper, sampled_data)
        shap_values = kernel_explainer(self.test_data.iloc[51:52, :])
        
        return shap_values

    def shap_plot(self):
        shap_values = self._calculate_shap_values()

        # Multiplying shap_values[0] by 1000 to scale the values for better visualization in the waterfall plot
        shap.plots.waterfall(shap_values[0] * 1000, show=False)
        explainer_params = f"background data size: {len(self.train_data)}, test instance index: 51"
        plt.title(f"KernelExplainer SHAP Waterfall Plot ({explainer_params})")
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
    neurons = [input_size] + [1024]*6 + [512, 256, 128]

    single_task_mlp = MultiTaskMLP(num_neurons=neurons, 
                                   droput_rate=0.0, 
                                   activation="relu", 
                                   output_neurons=[1])

    predictor = PredictionsExplainer(single_task_mlp, X_train, X_test)
    predictions = predictor._predict_wrapper(X_test)
    # print(predictions)
    # predictor.shap_plot()
