from features.dataset import XrayDataset
from chest_xray.features.cv import (
    XrayCV,
    get_normalization_stats,
    BATCH_SIZE, SEED, NUM_WORKERS, K_FOLDS
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

IMG_SIZE: int = 512


def main():
    print("Loading dataset and transforms")
    loader: XrayCV = XrayCV()
    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias = True),
        v2.ToDtype(torch.float32, scale = True)
    ])
    temp_ds: XrayDataset = XrayDataset(
        loader.train,
        transform = base_transform,
        diseases = loader.diseases
    )
    temp_loader: DataLoader = DataLoader(
        temp_ds, 
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS
    )
    mean, std = get_normalization_stats(temp_loader)
    print(f"Mean: {mean.item()}, Std: {std.item()}")
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias = True),
        v2.ColorJitter(brightness = 0.2, contrast = 0.2),
        v2.ToDtype(torch.float32, scale = True),
        v2.GaussianNoise(mean = 0.0, sigma = 0.02),
        v2.Normalize(mean = [mean.item()], std = [std.item()])
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias = True),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean = [mean.item()], std = [std.item()])
    ])
    folds: generator = loader.fold_loaders(
        k = K_FOLDS,
        batch_size = BATCH_SIZE,
        transform = train_transform
    )
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"Training Fold {fold_idx + 1} / {K_FOLDS}")
        val_loader.dataset.transform = val_transform


if __name__ == "__main__":
    main()

