import torch.nn as nn
from typing import Literal

class ActivationFactory:
    
    @staticmethod
    def activation(activation_name: 
                   Literal["relu", "tanh", "sigmoid"]) -> nn.Module:
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid": 
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")