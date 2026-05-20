from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
import numpy as np

PATCH_SIZE = 256


class DataClass(Dataset):
    def __init__(self, parquet_file, view_configuration: dict, split="train"):
        super().__init__()
        self.view_configuration = view_configuration
        self.split = split
        self.df = pd.read_parquet(parquet_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.preprocessors = {
            view_name: config_["preprocessor"]
            for view_name, config_ in view_configuration.items()
        }

    def __len__(self):
        return len(self.df)

    def _get_patch(self, image_: Image.Image, idx: int):
        width, height = image_.size
        patch_size = PATCH_SIZE
        if self.split in ["val", "test"]:
            x = (width - patch_size) // 2
            y = (height - patch_size) // 2
        else:
            seed = int(self.current_epoch * 997 + idx)
            rng = np.random.default_generator(np.random.PCG64(seed))

            max_x = width - self.patch_size
            max_y = height - self.patch_size

            x = rng.integers(0, max_x) if max_x > 0 else 0
            y = rng.integers(0, max_y) if max_y > 0 else 0

        return image_.crop((x, y, x + patch_size, y + patch_size))

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_ = Image.open(item["path"]).convert("RGB")
        image_ = self._get_patch(image_, idx)
        label = item["category"]

        view_outputs = {}
        for view_name, preprocessor in self.preprocessors.items():
            # this will run through all preprocessors, convert the outputs into a torch tensor and add the result back in a dictionary
            view_outputs[view_name] = torch.from_numpy(preprocessor(image_)).float()
        return {"views": view_outputs, "label": torch.tensor(label, dtype=torch.long)}
