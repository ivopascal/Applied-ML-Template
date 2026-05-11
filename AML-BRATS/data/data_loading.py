from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DATA_PATH = Path("data/BraTS2020_training_data")


def _process_path(path_str: str) -> Path:
    """Convert a single path to work with the project file structure."""
    relative_path = Path(path_str).relative_to("/")
    return DATA_PATH / relative_path


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the metadata and processess all paths."""
    metadata = pd.read_csv(path)
    metadata["slice_path"] = metadata["slice_path"].map(_process_path)
    return metadata


def split_by_volume(
    metadata: pd.DataFrame,
    seed: int = 42,
    train: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the metadata by volume (patient ID)."""
    volumes = metadata["volume"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(volumes)

    n_total = len(volumes)
    n_train = int(n_total * train)

    train_volumes = volumes[:n_train]
    test_volumes = volumes[n_train:]

    train_data = metadata[metadata["volume"].isin(train_volumes)]
    test_data = metadata[metadata["volume"].isin(test_volumes)]

    return train_data, test_data


class BRATSDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame) -> None:
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path = self.metadata.iloc[idx]["slice_path"]

        with h5py.File(path, "r") as f:
            image: np.ndarray = f["image"][()]
            mask: np.ndarray = f["mask"][()]

        return {
            "image": image,
            "mask": mask,
        }

        # Preprocessing and/or data augumentation should go here i think?


metadata = load_metadata(DATA_PATH / "content/data/meta_data.csv")
train_metadata, test_metadata = split_by_volume(metadata)

def make_cv_splits(
    train_metadata: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    #Splitting this with k cross-validation. (by patient)

    #to get all uniqur patient IDs from the training metadata
    volumes = train_metadata["volume"].unique()

    #random number generator so that we can have the same split every time we run
    rng = np.random.default_rng(seed)
    #to shuffle the patient ID
    rng.shuffle(volumes)

    #group the patients by splitting the shuffled ids
    volume_folds = np.array_split(volumes, k)

    cv_splits = []

    #go through each fold and use the current as a validation set while the rest are just training
    for i in range(k):
        val_volumes = volume_folds[i]

        train_volumes = np.concatenate([
            volume_folds[j] for j in range(k) if j != i
        ])
        #training folds
        fold_train = train_metadata[train_metadata["volume"].isin(train_volumes)]
        #validation folds
        fold_val = train_metadata[train_metadata["volume"].isin(val_volumes)]

        #save
        cv_splits.append((fold_train, fold_val))

    return cv_splits


metadata = load_metadata(DATA_PATH / "content/data/meta_data.csv")
train_metadata, test_metadata = split_by_volume(metadata)
cv_splits = make_cv_splits(train_metadata, k=5)
test_ds = BRATSDataset(test_metadata)
fold_number = 1

for fold_train_metadata, fold_val_metadata in cv_splits:
    print(f"Fold {fold_number}")
    fold_train_ds = BRATSDataset(fold_train_metadata)
    fold_val_ds = BRATSDataset(fold_val_metadata)
    print("Train size:", len(fold_train_ds))
    print("Validation size:", len(fold_val_ds))
    print()
    fold_number += 1

print("Test size:", len(test_ds))
print("One test example:")
print(test_ds[0])
