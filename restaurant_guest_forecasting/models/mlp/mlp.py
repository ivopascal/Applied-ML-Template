import torch
import torch.nn as nn
from typing import List, Literal, Tuple

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
        self.activation = activation

        self.model = MLPBase._model(num_neurons, droput_rate, activation)
        self.output_layers = self._output_layers()   
        

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

            modules_dict[f"linear_{i}"] = \
                          nn.Linear(in_neurons_i, out_neurons_i)
            modules_dict[f"{activation}_{i}"] = \
                          ActivationFactory.activation(activation_name=activation)
            modules_dict[f"dropout_{i}"] = \
                          nn.Dropout(droput_rate)
            
        model = nn.Sequential(modules_dict)
        return model
            
    @abstractmethod
    def _output_layers(self) -> Tuple[nn.Module]:
        """
        Abstract method for defining the final output layer.

        Must be implemented by subclasses.

        Note: Can be extended for more than two tasks.

        Returns:
            Tuple[nn.Module]: Output layers in case the model 
                            splits (for multitask-learning). For single task,
                            the tuple contains only one element.
        """
        pass

    def forward(self, x: torch.Tensor):
        """
        Runs a forward pass through the hidden layers and output layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        x = self.model(x)

        outputs = [output_layer(x) for output_layer in self.output_layers]

        return outputs
    

class SingleTaskMLP(MLPBase):
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu") \
                    -> None:
        
        super().__init__(num_neurons, droput_rate, activation)


    def _output_layers(self):
        # Single value predictiton
        output_layer = nn.Linear(self.num_neurons[-1], 1)
        return (output_layer,)
    

class MultiTaskSingleHeadMLP(MLPBase):
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu") \
                    -> None:
        
        super().__init__(num_neurons, droput_rate, activation)


    def _output_layers(self):
        # Predict both tasks at in one vector
        output_layer = nn.Linear(self.num_neurons[-1], 2)
        return (output_layer,)
    

class MultiTaskMultiHeadMLP(MLPBase):
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu") \
                    -> None:
        
        super().__init__(num_neurons, droput_rate, activation)


    def _output_layers(self):
        # Split the model into two output heads
        output_layer_task1 = nn.Linear(self.num_neurons[-1], 1)
        output_layer2 = nn.Linear(self.num_neurons[-1], 1)
        return (output_layer_task1, output_layer2)
