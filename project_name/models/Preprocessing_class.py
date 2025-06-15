import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from PIL import Image
import torch
from io import BytesIO


class Preprocessing:
    """
    Preprocessing class for preprocessing image and depth data.
    """
    def __init__(self, tile_size: Tuple[int, int] = (256, 256)) -> None:
        """
        Initialize the Preprocessing class with a tile size.
        """
        self.tile_size = tile_size
        self.last_padding_info: dict[int, dict] = {}

    def load_image(self, img: Union[str, Image.Image, BytesIO]) -> np.ndarray:
        """
        Load a PIL image, a path to image, or a BytesIO stream, and
        convert to numpy array (uint8).
        """
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, BytesIO):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image, "
                            "a BytesIO stream, or a path to one.")

        return np.array(img.convert("RGB"))

    def is_8_bit(self, np_array: np.ndarray) -> bool:
        """
        Check if a numpy array is of type uint8.
        """
        return np_array.dtype == np.uint8

    def normalize(self, np_array: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array to [0, 1].
        """
        if self.is_8_bit(np_array):
            normalized = np_array.astype(np.float32) / 255.0
        else:
            # Handle other types if necessary
            normalized = (np_array /
                          np.finfo(np_array.dtype).max).astype(np.float32)

        # Apply ImageNet mean and std (as used during training)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        return normalized

    def to_tensor(self, np_image: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy image (H, W, C) or (H, W) to a
        PyTorch tensor (C, H, W) or (1, H, W).
        Normalizes to float32.
        """
        norm = self.normalize(np_image)

        if norm.ndim == 2:  # grayscale
            return torch.from_numpy(norm).unsqueeze(0).float()
        elif norm.ndim == 3:  # RGB
            return torch.from_numpy(norm).permute(2, 0, 1).float()
        else:
            raise ValueError("Invalid input shape for tensor conversion")

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor (C, H, W) or
        (1, H, W) to a numpy array (H, W, C) or (H, W).
        Assumes tensor is already on CPU and detached.
        """
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                return tensor.squeeze(0).numpy()
            return tensor.permute(1, 2, 0).numpy()
        elif tensor.ndim == 2:
            return tensor.numpy()
        else:
            raise ValueError("Invalid tensor shape for numpy conversion")

    def tile_with_padding(self,
                          np_arrays: list[np.ndarray],
                          pad_mode: str = 'constant') -> np.ndarray:
        """Split input array into tiles and pad if part is smaller than
        the tile

        Args:
            np_arrays (list[np.ndarray]): ndarray of image
            pad_mode (str, optional): mode of padding. Defaults to 'constant'.

        Raises:
            TypeError: if given type is not a ndarray

        Returns:
            _type_: array of tiles
        """
        if not isinstance(np_arrays, (list, tuple)):
            np_arrays = [np_arrays]

        all_tiles = []
        for idx, np_array in enumerate(np_arrays):
            if not isinstance(np_array, np.ndarray):
                raise TypeError("Input must be numpy array")
            original_shape = np_array.shape
            tile_h, tile_w = self.tile_size
            h, w = original_shape[:2]

            # Store padding info before any processing
            self.last_padding_info[idx] = {
                'original_shape': original_shape,
                'pad_h': (tile_h - (h % tile_h)) % tile_h,
                'pad_w': (tile_w - (w % tile_w)) % tile_w,
                'is_grayscale': len(original_shape) == 2
            }
            pad_h = self.last_padding_info[idx]['pad_h']
            pad_w = self.last_padding_info[idx]['pad_w']

            # Pad before normalization
            if len(original_shape) == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            else:
                pad_width = ((0, pad_h), (0, pad_w))

            padded = np.pad(np_array, pad_width, mode=pad_mode)
            tiles = []
            for i in range(0, padded.shape[0], tile_h):
                for j in range(0, padded.shape[1], tile_w):
                    tile = padded[i:i + tile_h, j:j + tile_w]
                    if tile.shape[:2] != (tile_h, tile_w):
                        tile = np.pad(tile,
                                      ((0, tile_h - tile.shape[0]),
                                       (0, tile_w - tile.shape[1])),
                                      mode=pad_mode)
                    tiles.append(tile)

            all_tiles.extend(tiles)

        return np.array(all_tiles)

    def reconstruct_depth(self, depth_tiles: np.ndarray,
                          original_idx: int = 0) -> np.ndarray:
        """
        Special reconstruction for 1-channel depth outputs
        """
        info = self.last_padding_info[original_idx]
        h, w = info['original_shape'][:2]
        tile_h, tile_w = self.tile_size

        # Create padded canvas
        padded_h = h + info['pad_h']
        padded_w = w + info['pad_w']
        reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
        cols = padded_w // tile_w
        rows = padded_h // tile_h

        # Reconstruct depth map
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(depth_tiles):
                    break
                y_start = i * tile_h
                y_end = y_start + tile_h
                x_start = j * tile_w
                x_end = x_start + tile_w
                tile = depth_tiles[idx]
                actual_h = min(tile_h, padded_h - y_start)
                actual_w = min(tile_w, padded_w - x_start)
                reconstructed[y_start:y_end, x_start:x_end] =\
                    tile[:actual_h, :actual_w]

        # Crop to original dimensions
        return reconstructed[:h, :w]

    def depth_to_rgb(self,
                     depth_map: np.ndarray,
                     cmap: str = 'plasma',
                     invert: bool = False) -> np.ndarray:
        """
        Depth map to RGB.
        """
        # Normalize based on percentiles (robust to outliers)
        p1, p99 = np.percentile(depth_map, [1, 99])
        scaled = np.clip((depth_map - p1) / (p99 - p1), 0, 1)

        if invert:
            scaled = 1 - scaled

        cmap = plt.get_cmap(cmap)
        return (cmap(scaled)[..., :3] * 255).astype(np.uint8)
