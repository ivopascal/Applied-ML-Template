from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image

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

    def get_patch(self, img: Image.Image):
        width, height = img.size
        patch_size = PATCH_SIZE
        x = (width - patch_size) // 2
        y = (height - patch_size) // 2
        return img.crop((x, y, x + patch_size, y + patch_size))

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_ = Image.open(item["path"]).convert("RGB")
        label = item["category"]

        view_outputs = {}
        for view_name, preprocessor in self.preprocessors.items():
            # this will run through all preprocessors, convert the outputs into a torch tensor and add the result back in a dictionary
            view_outputs[view_name] = torch.from_numpy(preprocessor(image_)).float()
        return {"views": view_outputs, "label": torch.tensor(label, dtype=torch.long)}
