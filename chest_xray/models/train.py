import pandas as pd
import torch
import wandb
from ..features.cv import XrayCV
from collections.abc import Iterator
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from chest_xray.data.chestdataset import ChestXRayDataset
from chest_xray.data.labels import CLASSES


BATCH_SIZE: int = 4  # play around with this on Habrok
SEED: int = 42
NUM_WORKERS: int = 4    # play around with this on Habrok
K_FOLDS: int = 4


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.root = Path(__file__).parent.parent.parent
        self.image_root = self.root / "data" / "images"
        self.classes = CLASSES
        self.batch_size = BATCH_SIZE
        self.image_size = 512
        self.seed = SEED
        self.cv = XrayCV(BATCH_SIZE, SEED, NUM_WORKERS, K_FOLDS)


    def transform_images(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])
        

    def yield_dataloaders(self, transform) -> Iterator[tuple[DataLoader, ...]]:
        for train_loader, val_loader in self.cv.fold_loaders(transform):
            yield train_loader, val_loader

        
    def train_one_epoch(self, epoch, num_epochs, phase, dataloader):
        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Train",
            unit="batch",
            dynamic_ncols=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / batch_idx

            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)


    def validate_one_epoch(self, epoch, num_epochs, phase, dataloader):
        self.model.eval()

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
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                average_loss = running_loss / batch_idx

                progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)
    

    def train(self, num_epochs, train_loader, val_loader):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch, num_epochs, "Phase 1", train_loader)
            val_loss = self.validate_one_epoch(epoch, num_epochs, "Phase 1", val_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        return train_loss, val_loss
