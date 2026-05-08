from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import timm


class LayerTemplate(nn.Module, ABC):
    def __init__(self, embed_dim = 512):
        super().__init__()
        self.embed_dim = embed_dim
        


    @abstractmethod
    def model_initialisation(self):
        """Subclasses will initialise their own layers."""
        pass


    def forward(self, x):
        return self.model(x)


class MLPLayer(LayerTemplate):
    def __init__(self, input_size, hidden_dim = None):
        self.input_size = input_size
        self.hidden_dim = hidden_dim or [256]

        super().__init__()

        self.model = self.model_initialisation()


    def model_initialisation(self):
        layers = [nn.Flatten()]
        curr = self.input_size
        for h in self.hidden_dim:
            layers.extend([nn.Linear(curr, h), nn.ReLU()])
            curr = h
        layers.append(nn.Linear(curr, self.embed_dim))
        return nn.Sequential(*layers)


class CNNLayer(LayerTemplate):
    def __init__(self, arch = 'resnet18', input_chan = 3):
        self.arch = arch
        self.input_chan = input_chan

        super().__init__()

        self.model = self.model_initialisation()


    def model_initialisation(self):
        cnn = timm.create_model(
            self.arch,
            pretrained=True,
            num_classes=0,
            in_chans=self.input_chan)
        
        n_features = int(cnn.num_features)

        return nn.Sequential(cnn, nn.Linear(n_features, self.embed_dim))