from torchvision import models
from timm.loss import AsymmetricLossMultiLabel
import torch
import wandb
from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES
from ..tools.globals import *

# https://deepwiki.com/andreasveit/densenet-pytorch/6.2-training-configuration
# 161 model has the best top-1 and top-5 accuracy 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet = models.densenet161(weights=None)
densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, len(CLASSES))

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
densenet = densenet.to(device)

criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05)
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

modelTrainer = ModelTrainer(densenet, criterion, optimizer, device)



def trainModel():
    transform = modelTrainer.transform_images(modelTrainer.image_size)

    for fold_idx, (train_loader, val_loader) in enumerate(
        modelTrainer.yield_dataloaders(transform)
    ):
        print(f"\nFold {fold_idx + 1}/{K_FOLDS}")

        run = wandb.init(
            project="chest_xray_cv",
            entity="chest_xray",
            name=f"fold_{fold_idx+1}",
            config={
                "fold": fold_idx + 1,
                "epochs": 10,
                "batch_size": BATCH_SIZE,
                "lr": 1e-3,
                "model": "densenet161",
            },
        )

        densenet = models.densenet161(weights=None)
        densenet.classifier = torch.nn.Linear(
            densenet.classifier.in_features,
            len(CLASSES),
        )

        old_first_layer = densenet.features[0]
        new_first_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_first_layer.out_channels,
            kernel_size=old_first_layer.kernel_size,
            stride=old_first_layer.stride,
            padding=old_first_layer.padding,
        )

        with torch.no_grad():
            new_first_layer.weight = torch.nn.Parameter(
                old_first_layer.weight.mean(dim=1, keepdim=True)
            )
            new_first_layer.bias = old_first_layer.bias

        densenet.features[0] = new_first_layer
        densenet = densenet.to(device)

        criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05)
        optimizer = torch.optim.Adam(densenet.parameters(), lr=1e-3)

        trainer = ModelTrainer(densenet, criterion, optimizer, device)

        for epoch in range(10):
            train_loss = trainer.train_one_epoch(
                epoch, 10, f"Fold {fold_idx+1}", train_loader
            )

            val_loss = trainer.validate_one_epoch(
                epoch, 10, f"Fold {fold_idx+1}", val_loader
            )

            # 🔥 log to wandb
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            print(
                f"Fold {fold_idx+1} Epoch {epoch+1}: "
                f"Train {train_loss:.4f} | Val {val_loss:.4f}"
            )

        run.finish()

if __name__ == "__main__":
    trainModel()