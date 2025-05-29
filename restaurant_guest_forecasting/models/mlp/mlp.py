import torch
import torch.nn as nn
from typing import List, Literal

from restaurant_guest_forecasting.models.mlp.activation_factory import\
      ActivationFactory

from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu",
                 output_neurons:int = 1) \
                    -> None:
        """
        Initializes the MLP.

        Args:
            num_neurons (List[int]): List of neurons per layer including input,
                                     but excluding output layer.
            droput_rate (float): Dropout rate between layers (default is 0.0).
            activation (str): Activation function ("relu", "tanh", "sigmoid").
            output_neurons (int): Number of neurons in the final output layer.
        """
        super().__init__()
        self.model = MLP._model()
        self.output_layer = nn.Linear(num_neurons[-1], output_neurons)                       

    @staticmethod
    def _model(num_neurons: List[int], 
               droput_rate: float = 0.0,
               activation: str = Literal["relu", "tanh", "sigmoid"] = "relu")\
         -> nn.Sequential:
        """
        Constructs the sequential MLP model.

        Args:
            num_neurons (List[int]): List of neurons per layer.
            droput_rate (float): Dropout rate between layers.
            activation (str): Activation function to apply between layers.

        Returns:
            nn.Sequential: The MLP model.
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
            


    def forward(self, x: torch.Tensor):
            """
            Forward pass of the MLP.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_features).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_neurons).
            """
            x = self.model(x)
            return self.output_layer(x)