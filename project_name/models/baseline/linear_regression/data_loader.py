from typing import Tuple, Any
import numpy as np
import cv2  # type: ignore
from data.data_loader import DataLoader


class LinearRegressionDataset:
    """Dataset loader for Linear Regression training & testing data."""

    def __init__(self,
                 split: str,
                 tile_size: Tuple[int, int] = (64, 64)) -> None:
        """
        Args:
            split (str): "train" or "val"
            tile_size (tuple): (height, width) of tiles
        """
        mapped_split = "Val" if split.lower() == "val" else "train"
        self.data_loader = DataLoader(mapped_split)

        self.X = []
        self.y = []
        self.original_images = []
        self.original_depth_maps = []

        resize_dim = (384, 512)
        tile_h, tile_w = tile_size

        for _ in range(len(self.data_loader.data_paths)):
            image, depth = self.data_loader.get_data()

            i = cv2.INTER_AREA
            image_small = cv2.resize(image, resize_dim, i)
            depth_small = cv2.resize(depth, resize_dim, i)

            self.original_images.append(image_small)
            self.original_depth_maps.append(depth_small)

            img_h, img_w = image_small.shape[:2]

            assert img_h % tile_h == 0 and img_w % tile_w == 0, \
                f"Image {img_h}x{img_w} not divisible by {tile_h}x{tile_w}"

            for y in range(0, img_h, tile_h):
                for x in range(0, img_w, tile_w):
                    img_tile = image_small[y:y + tile_h, x:x + tile_w]
                    depth_tile = depth_small[y:y + tile_h, x:x + tile_w]

                    self.X.append(img_tile.flatten())
                    self.y.append(depth_tile.flatten())

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self) -> int:
        """Return the length of the dataset (number of tiles)."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return an item at a specified index."""
        return self.X[idx], self.y[idx]

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return full dataset as (X, y) numpy arrays."""
        return self.X, self.y
