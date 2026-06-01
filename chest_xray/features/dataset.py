import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

#   ▖▖          ▄   ▗       ▗
#   ▚▘▄▖▛▘▀▌▌▌  ▌▌▀▌▜▘▀▌▛▘█▌▜▘
#   ▌▌  ▌ █▌▙▌  ▙▘█▌▐▖█▌▄▌▙▖▐▖
#           ▄▌

class XrayDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform = None, 
        diseases = None
    ) -> None:
        self.data: pd.DataFrame = dataframe.reset_index(drop = True)
        self.transform = transform
        if diseases is None:
            self.diseases: list[str] = sorted(
                self.data["diseases"].str.split("|").explode().unique()
            )
        else:
            self.diseases: list[str] = diseases
        self.disease_to_idx: dict[str, int] = {
            disease: i for i, disease in enumerate(self.diseases)
        }


    def __len__(self) -> int:
        """Returns amount of samples in the dataset"""
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple[Image, torch.Tensor]:
        """Get item from dataset with dataset[n]"""
        row: pd.Series = self.data.iloc[idx]
        image: Image = Image.open(row["img_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        target: torch.Tensor = torch.zeros(len(self.diseases))
        for disease in row["diseases"].split("|"):
            if disease in self.disease_to_idx:
                target[self.disease_to_idx[disease]] = 1.0
        return image, target

