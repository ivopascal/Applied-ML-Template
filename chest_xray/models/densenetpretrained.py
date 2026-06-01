from torchvision import models
from timm.loss import AsymmetricLossMultiLabel
import torch
from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES

# https://deepwiki.com/andreasveit/densenet-pytorch/6.2-training-configuration
# 161 model has the best top-1 and top-5 accuracy 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

densenet = models.densenet161(weights=models.DenseNet161_Weights)
densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, len(CLASSES))
densenet = densenet.to(device)

criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05)
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

# take first layer
old_first_layer = densenet.features[0]

# Create new convolution layer with 1 input channel and keep the rest the same
new_first_layer = torch.nn.Conv2d(
    in_channels=1,
    out_channels=old_first_layer.out_channels,
    kernel_size=old_first_layer.kernel_size,
    stride=old_first_layer.stride,
    padding=old_first_layer.padding,
)

# Initialize the new first layer's weights by averaging the weights of the original 3 channels
with torch.no_grad():
    new_first_layer.weight = torch.nn.Parameter(
        old_first_layer.weight.mean(dim=1, keepdim=True)
    )
    new_first_layer.bias = old_first_layer.bias

densenet.features[0] = new_first_layer

modelTrainer = ModelTrainer(densenet, criterion, optimizer, device)

def trainModel():
    train_loader, val_loader = modelTrainer.load_data()
    modelTrainer.train(num_epochs=10, train_loader=train_loader, val_loader=val_loader)
    

if __name__ == "__main__":
    trainModel()