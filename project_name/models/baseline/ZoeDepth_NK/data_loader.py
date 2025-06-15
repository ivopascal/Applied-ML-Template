import torch  # type: ignore
from torchvision.transforms import functional as TF  # type: ignore
from typing import Tuple
from torch.utils.data import Dataset  # type: ignore
from data.data_loader import DataLoader as OriginalDataLoader


class ZoeDepthDataset(Dataset):
    """Zoedepth dataloader

    Args:
        Dataset (Dataset): Base dataset object
    """
    def __init__(self, split: str) -> None:
        """Init dataset"""
        if split.lower() not in {"train", "val"}:
            raise ValueError("Split must be 'train' or 'val'")
        self.loader = OriginalDataLoader(split.lower())
        self.total_samples = len(self.loader.data_paths)

    def __len__(self) -> int:
        """get amount samples"""
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Item magic method"""
        self.loader.data_index = idx
        image_np, depth_np = self.loader.get_data()

        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        depth_np = depth_np.squeeze(-1)

        depth = torch.from_numpy(depth_np).float()
        depth = depth.unsqueeze(0)
        depth = TF.resize(
            depth,
            [384, 512],
            interpolation=TF.InterpolationMode.BILINEAR
        )
        depth = depth.squeeze(0)

        depth = depth.unsqueeze(0)

        return image, depth
