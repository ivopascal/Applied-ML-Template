import os

import torch
import torch.nn as nn
from timm.loss import AsymmetricLossMultiLabel
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

from chest_xray.features.loader import XrayLoader
from chest_xray.features.transforms import get_vgg16_transform
    
# Training function
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    phase: str,
) -> float:
    """Train the model for one epoch"""
    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Train",
        unit="batch",
        dynamic_ncols=True
    )

    for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        average_loss = running_loss / batch_idx

        progress_bar.set_postfix(loss=f"{average_loss:.4f}")

    return running_loss / len(dataloader)

# Validation function
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    phase: str,
) -> float:
    """Validate the model for one epoch"""
    model.eval()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Val",
        unit="batch",
        dynamic_ncols=True
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            average_loss = running_loss / batch_idx

            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

    return running_loss / len(dataloader)

# The main function that will run the training and validation loops, and save the model at the end
def main() -> None:
    """Loads the dataset, creates train and validation splits, initializes the
    VGG16 model, trains the classifier head, fine-tunes the full model, and
    saves the trained model checkpoint."""

    # Using GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 16
    num_epochs = 30
    head_epochs = 5
    head_lr = 1e-3
    fine_tune_lr = 1e-4
    image_size = 512
    k_folds = 4

    # The transforms of the images for training and validation
    transform = get_vgg16_transform(image_size)

    xray_loader = XrayLoader()
    classes = xray_loader.diseases

    train_loader, val_loader = next(
        xray_loader.fold_loaders(
            k=k_folds,
            batch_size=batch_size,
            transform=transform,
        )
    )

    print(f"Actual train dataset size: {len(train_loader.dataset)}")
    print(f"Actual validation dataset size: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(classes)}")

    # Model
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)

    # Change VGG16 from 3-channel input to 1-channel input
    old_first_layer = model.features[0]

    # Create new cnn layer with 1 input channel and keep the rest the same
    new_first_layer = nn.Conv2d(
        in_channels=1,
        out_channels=old_first_layer.out_channels,
        kernel_size=old_first_layer.kernel_size,
        stride=old_first_layer.stride,
        padding=old_first_layer.padding,
    )

    # Initialize the new first layer's weights by averaging the weights of the original 3 channels
    with torch.no_grad():
        new_first_layer.weight = nn.Parameter(
            old_first_layer.weight.mean(dim=1, keepdim=True)
        )
        new_first_layer.bias = old_first_layer.bias

    model.features[0] = new_first_layer

    # Change final classifier output to 15 classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes))

    # Freeze convolutional feature extractor for head-only training
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)

    # Loss
    criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05)

    # Optimizer for head-only training
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_lr,
    )

    # Phase 1: Train classifier head
    print("\nStarting head-only training...\n")

    for epoch in range(head_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, "Head only")
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs, "Head only")

        print(f"Epoch [{epoch + 1}/{num_epochs}] "f"Phase: Head only "f"Train Loss: {train_loss:.4f} "f"Val Loss: {val_loss:.4f}")

    # Phase 2: Fine-tune full model
    print("\nStarting fine-tuning...\n")

    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)

    for epoch in range(head_epochs, num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, "Fine-tuning")
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs, "Fine-tuning")

        print(f"Epoch [{epoch + 1}/{num_epochs}] "f"Phase: Fine-tuning "f"Train Loss: {train_loss:.4f} "f"Val Loss: {val_loss:.4f}")

    # Save model
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, "vgg16_chest_xray_asl.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": image_size,
            "num_epochs": num_epochs,
            "head_epochs": head_epochs,
            "head_lr": head_lr,
            "fine_tune_lr": fine_tune_lr,
        },
        save_path,
    )

    print(f"\nModel saved as {save_path}")

if __name__ == "__main__":
    main()