from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import Tuple

from ..models.cnn import CNNBackbone
from ..models.Preprocessing_class import Preprocessing

# Locate your Data folder
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


class CNNDataset(Dataset):
    """Class to load and preprocess images and depth maps for CNN training."""
    def __init__(
            self,
            split: str,
            tile_size: Tuple[int, int] = (256, 256)) -> None:
        """
        split: 'train_subset' or 'val_subset' folder under Data/
        """
        folder = DATA_DIR / split
        self.samples = sorted(d for d in folder.iterdir() if d.is_dir())
        preprocessing = Preprocessing(tile_size)
        self.tileer = preprocessing.tile_with_padding
        self.normalizer = preprocessing.normalize
        self.tile_h, self.tile_w = tile_size

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns a tuple (image, depth) for the given index."""
        sample_dir = self.samples[idx]
        image_path = next(sample_dir.glob("*.png"))
        depth_path = next(sample_dir.glob("*_depth.npy"))

        image = np.array(Image.open(image_path))
        depth = np.load(depth_path).astype(np.float32)

        # 1. Tile & normalize image as before
        img_tile = self.tileer(image)[0]

        # 2. Normalize depth to [0,1]
        depth_norm = self.normalizer(depth)

        # Print debugging information for shape
        # print(f"Depth shape before normalization: {depth.shape}")
        # print(f"Depth shape after normalization: {depth_norm.shape}")

        # Ensure depth_norm is 2D
        if depth_norm.ndim > 2:
            depth_norm = np.squeeze(depth_norm)
            if depth_norm.ndim > 2:
                depth_norm = depth_norm[:, :, 0]

        h, w = depth_norm.shape

        # 3. Pad depth so dims % tile == 0
        pad_h = (self.tile_h - (h % self.tile_h)) % self.tile_h
        pad_w = (self.tile_w - (w % self.tile_w)) % self.tile_w

        depth_padded = np.pad(
            depth_norm,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
        )

        # 4. Extract first tile
        depth_tile = depth_padded[: self.tile_h, : self.tile_w]

        # 5. Convert to tensors
        x = torch.from_numpy(img_tile).permute(2, 0, 1).float()
        y = torch.from_numpy(depth_tile).unsqueeze(0).float()

        return x, y


def train_cnn(
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_epochs: int
) -> None:
    """Method to train the CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training CNN: epochs={epochs}, "
        f"bs={batch_size}, lr={lr}, freeze={freeze_epochs}"
    )

    model = CNNBackbone(pretrained=True).to(device)

    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        [
            {"params": model.head.parameters(), "lr": lr},
            {"params": model.backbone.parameters(), "lr": lr * 0.1},
        ]
    )
    loss_fn = nn.MSELoss()

    train_ds = CNNDataset("train_subset")
    val_ds = CNNDataset("val_subset")
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    best_val = float("inf")
    best_ckpt = None

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("Backbone unfrozen")

        # Training
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_dl)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                total_val_loss += loss_fn(model(xb), yb).item()
        val_loss = total_val_loss / len(val_dl)

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = model.state_dict().copy()

    # Restore best & save
    model.load_state_dict(best_ckpt)
    output_path = BASE_DIR / "cnn_best.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Saved best model to {output_path}")
