from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import timm


class LayerTemplate(nn.Module, ABC):
    """
    This is a template for the layers that will be used in the multi-view architecture.
    It defines the basic structure and the forward method.
    """
    def __init__(self,) -> None:
        """
        Initialize the layer template.
        """
        super().__init__()


    @abstractmethod
    def model_initialisation(self) -> nn.Sequential:
        """Subclasses will initialise their own layers."""
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        return self.model(x)


class MLPLayer(LayerTemplate):
    """
    A simple MLP layer that can be used for tabular data or any other data that can be flattened.
    """
    def __init__(self, input_size : int, hidden_dim : list = [256], embed_dim : int = 512) -> None:
        """
        Initialize the MLP layer.
        Args:
            input_size (int): The size of the input features.
            hidden_dim (list): A list of hidden layer dimensions. Defaults to [256].
            embed_dim (int): The dimension of the output embedding. Defaults to 512.
        """
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        super().__init__()

        self.model = self.model_initialisation()


    def model_initialisation(self) -> nn.Sequential:
        """
        Initialize the MLP model with the specified input size, hidden dimensions,
        and output embedding dimension.
        Returns:
            nn.Sequential: The initialized MLP model.
        """
        layers = [nn.Flatten()]
        curr = self.input_size
        
        for h in self.hidden_dim:
            layers.extend([nn.Linear(curr, h), nn.ReLU()])
            curr = h
        layers.append(nn.Linear(curr, self.embed_dim))

        return nn.Sequential(*layers)


class CNNLayer(LayerTemplate):
    """
    A CNN layer that can be used for image data. It uses a pre-trained model
    from timm and adds a linear layer at the end to get the desired embedding dimension.
    """
    def __init__(self, arch: str = 'resnet18', input_chan: int = 3, embed_dim : int = 512) -> None:
        self.arch = arch
        self.input_chan = input_chan
        self.embed_dim = embed_dim

        super().__init__()

        self.model = self.model_initialisation()


    def model_initialisation(self) -> nn.Sequential:
        """
        Initialize the CNN model using a pre-trained architecture.
        Returns:
            nn.Sequential: The initialized CNN model.
        """
        cnn = timm.create_model(
            self.arch,
            pretrained=True,
            num_classes=0,
            in_chans=self.input_chan)
        
        n_features = int(cnn.num_features)

        return nn.Sequential(cnn, nn.Linear(n_features, self.embed_dim))
