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
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu",
                 dtype=torch.float64) \
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

        self.model = MLPBase._model(num_neurons, droput_rate, activation, dtype=dtype)
        self.output_layers = nn.ModuleList(self._output_layers()) 
        self.dtype = dtype
        

    @staticmethod
    def _model(num_neurons: List[int], 
               droput_rate: float = 0.0,
               activation: Literal["relu", "tanh", "sigmoid"] = "relu",
               dtype=torch.float64)\
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
                          nn.Linear(in_neurons_i, out_neurons_i, dtype=dtype)
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
    
    

class MultiTaskMLP(MLPBase):
    """
    General-purpose MLP model supporting both single-task and multi-task learning.

    This model can handle:
        - Single-output regression/classification (e.g., output_neurons = [1])
        - Multi-output regression/classification with multiple heads (e.g., [1, 1], [2, 3])
    """
    def __init__(self,
                 num_neurons: List[int],
                 droput_rate: float = 0.0,
                 activation: Literal["relu", "tanh", "sigmoid"] = "relu",
                 output_neurons: List[int] = [1],
                 dtype=torch.float64) \
                    -> None:
        """
        Initializes the MultiTaskMLP model.

        Args:
            num_neurons (List[int]): List of neuron counts per hidden layer, including input.
                                The length also determines the depth of the network.
                                    Example: [10, 64, 32]
            droput_rate (float): Dropout probability to apply after each hidden layer (default: 0.0).
            activation (str): Activation function to use in hidden layers. One of: "relu", "tanh", "sigmoid".
            output_neurons (List[int]): A list where each element defines the output size for one task.
                                    Example:
                                          - [1]    -> single-task regression
                                          - [1, 1] -> two-task regression (two heads)
        """
        self.output_neurons = output_neurons
        self.dtype = dtype
        super().__init__(num_neurons, droput_rate, activation, dtype=dtype)
        


    def _output_layers(self):
        """
        Creates one output head per task using `output_neurons`.

        Returns:
            Tuple[nn.Module]: Tuple of output layers, one per task.
        """
        output_layers = [nn.Linear(self.num_neurons[-1], out_neurons, dtype=self.dtype) \
                         for out_neurons in self.output_neurons]
        return tuple(output_layers)
