import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader,\
                                                    guest_df_to_tensor_dataset

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP

import numpy as np

class ExplainPredictions():
    def __init__(self, 
                 model: nn.Module,
                 train_data: DataLoader,
                 test_data: DataLoader):
        
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def predict_wrapper(self, device='cpu'):
        self.model.to(device)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_data:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = self.model(X_batch)
                if isinstance(outputs, list):
                    outputs = outputs[0] 
                outputs = outputs.detach().cpu().numpy()
                predictions.append(outputs)
        
        predictions = np.concatenate(predictions, axis=0).flatten()
        print(predictions.shape)

    def calculate_shap_values(self):
        pass

    def shap_plot(self):
        pass

if __name__ == "__main__":
    train_df, _, test_df = train_val_test_data()

    # Training DataLoader
    train_loader = prepare_dataloader(df=train_df,
                                      batch_size=64, 
                                      to_tensor_fn=guest_df_to_tensor_dataset,
                                      is_train=True)
    # Test DataLoader
    test_loader   = prepare_dataloader(df=test_df,
                                      batch_size=64,
                                      to_tensor_fn=guest_df_to_tensor_dataset,
                                      is_train=False)
    
    input_size = 37
    neurons = [input_size, input_size, input_size]

    single_task_mlp = MultiTaskMLP(num_neurons=neurons, 
                                   droput_rate=0.0, 
                                   activation="relu", 
                                   output_neurons=[1])

    predictor = ExplainPredictions(single_task_mlp, train_loader, test_loader)
    predictions = predictor.predict_wrapper()
