import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ExplainPredictions():
    def __init__(self, 
                 model: nn.Module,
                 train_data: DataLoader,
                 test_data: DataLoader):
        
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def predict_wrapper(self):
        pass

    def calculate_shap_values(self):
        pass

    def shap_plot(self):
        pass
