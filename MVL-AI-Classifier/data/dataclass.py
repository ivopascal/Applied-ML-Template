from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

PATCH_SIZE = 256


class DataClass(Dataset):
    def __init__(self, parquet_file, split="train"):
        self.df = pd.read_parquet(parquet_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def get_patch(self, img: Image.Image):
        width, height = img.size
        patch_size = PATCH_SIZE
        x = (width - patch_size) // 2
        y = (height - patch_size) // 2
        return img.crop((x, y, x + patch_size, y + patch_size))

    def __getitem__(self, idx):
        # Need to change this, Pytorch expects different format, needs a tensor so we need to call preprocessors from here, but that requires the setup of the view factory structure first.
        item = self.df.iloc[idx]
        img = Image.open(item["path"]).convert("RGB")
        return self.get_patch(img)
