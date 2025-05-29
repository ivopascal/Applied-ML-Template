import torch
import torch.nn as nn
from typing import List, Literal

from restaurant_guest_forecasting.models.mlp.activation_factory import\
      ActivationFactory

from collections import OrderedDict

from abc import ABC, abstractmethod


class MLPBase(ABC, nn.Module):
    """
    Abstract base class for MLP models, providing shared architecture construction
    and requiring subclasses to define their own output layers.
    """
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu") \
                    -> None:
        """
        Initializes the core MLP structure (excluding the output layer).

        Args:
            num_neurons (List[int]): List of neurons per layer including input,
                                     but excluding output layer.
            droput_rate (float): Dropout rate between layers (default is 0.0).
            activation (str): Activation function ("relu", "tanh", "sigmoid").
        """
        super().__init__()

        self.num_neurons = num_neurons
        self.dropout_rate = droput_rate
        self.actication = activation

        self.model = MLPBase._model(num_neurons, droput_rate, activation)
        self.output_layer = MLPBase._output_layer()                       

    @staticmethod
    def _model(num_neurons: List[int], 
               droput_rate: float = 0.0,
               activation: Literal["relu", "tanh", "sigmoid"] = "relu")\
         -> nn.Sequential:
        """
        Constructs the sequential MLP model, i.e. input & hidden layers.

        Args:
            num_neurons (List[int]): List of neurons per layer.
            droput_rate (float): Dropout rate between layers.
            activation (str): Activation function to apply between layers.

        Returns:
            nn.Sequential: The MLP model excluding the output layers.
        """
        modules_dict = OrderedDict()   # Store the modules to initialize Sequential     
        
        for i in range(len(num_neurons) - 1):
            in_neurons_i = num_neurons[i]
            out_neurons_i = num_neurons[i + 1]

            modules_dict[f"linear_{i}",
                          nn.Linear(in_neurons_i, out_neurons_i)]
            modules_dict[f"{activation}_{i}",
                          ActivationFactory.activation(activation_name=activation)]
            modules_dict[f"dropout_{i}",
                          nn.Dropout(droput_rate)]
            
        model = nn.Sequential(modules_dict)
        return model
            
    @abstractmethod
    def _output_layer(self) -> nn.Module:
        """
        Abstract method for defining the final output layer.

        Must be implemented by subclasses.

        Returns:
            nn.Module: The output layer (e.g., Linear, Softmax, etc.).
        """
        pass

    def forward(self, x):
        """
        Runs a forward pass through the hidden layers and output layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        x = self.model(x)
        return self.output_layer(x)