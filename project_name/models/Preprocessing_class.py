import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, List


class Preprocessing:
    def __init__(self, tile_size: Tuple[int, int] = (256, 256)) -> None:
        """
        Initialize the Preprocessing class with a tile size.
        """
        self.tile_size = tile_size
        self.last_padding_info: dict[int, dict] = {}

    def is_8_bit(self, np_array: np.ndarray) -> bool:
        """
        Check if a numpy array is of type uint8.
        """
        return np_array.dtype == np.uint8

    def normalize(self, np_array: np.ndarray) -> np.ndarray:
        """
        Normalize an array to the [0, 1] range as float32.
        """
        if self.is_8_bit(np_array):
            return np_array.astype(np.float32) / 255.0
        if np.issubdtype(np_array.dtype, np.floating):
            return np_array.astype(np.float32)
        return (np_array / np.iinfo(np_array.dtype).max).astype(np.float32)

    def tile_with_padding(
        self,
        np_arrays: Union[np.ndarray, List[np.ndarray]],
        pad_mode: str = 'constant'
    ) -> np.ndarray:
        """
        Tile one or more images with padding to fit the specified tile size.
        """
        if not isinstance(np_arrays, (list, tuple)):
            np_arrays = [np_arrays]

        all_tiles = []
        for idx, np_array in enumerate(np_arrays):
            if not isinstance(np_array, np.ndarray):
                raise TypeError("Input must be numpy array")
            if not self.is_8_bit(np_array):
                raise ValueError("Input must be uint8 array")

            original_shape = np_array.shape
            normalized = self.normalize(np_array.copy())

            tile_h, tile_w = self.tile_size
            h, w = original_shape[:2]

            pad_h = (tile_h - (h % tile_h)) % tile_h
            pad_w = (tile_w - (w % tile_w)) % tile_w

            self.last_padding_info[idx] = {
                'original_shape': original_shape,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'is_grayscale': len(original_shape) == 2
            }

            if len(original_shape) == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            else:
                pad_width = ((0, pad_h), (0, pad_w))

            padded = np.pad(normalized, pad_width, mode=pad_mode)
            padded_h, padded_w = padded.shape[:2]

            tiles = []
            for i in range(0, padded_h, tile_h):
                for j in range(0, padded_w, tile_w):
                    tile = padded[i:i + tile_h, j:j + tile_w]
                    if tile.shape[:2] != (tile_h, tile_w):
                        tile = np.pad(
                            tile,
                            ((0, tile_h - tile.shape[0]),
                             (0, tile_w - tile.shape[1])),
                            mode=pad_mode
                        )
                    tiles.append(tile)
            all_tiles.extend(tiles)

        return np.array(all_tiles)

    def reconstruct_image(
        self,
        tiles: np.ndarray,
        original_idx: int = 0
    ) -> np.ndarray:
        """
        Reconstruct the original image from tiles.
        """
        info = self.last_padding_info.get(original_idx)
        if not info:
            raise ValueError("No padding info found for this index")

        tile_h, tile_w = self.tile_size
        h, w = info['original_shape'][:2]
        pad_h = info['pad_h']
        pad_w = info['pad_w']

        rows = (h + pad_h) // tile_h
        cols = (w + pad_w) // tile_w

        if info['is_grayscale']:
            recon_shape = (h + pad_h, w + pad_w)
        else:
            recon_shape = (h + pad_h, w + pad_w, info['original_shape'][2])

        reconstructed = np.zeros(recon_shape, dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                reconstructed[
                    i * tile_h:(i + 1) * tile_h,
                    j * tile_w:(j + 1) * tile_w
                ] = tiles[idx]

        final_image = np.clip(reconstructed[:h, :w], 0, 1)
        if info['is_grayscale']:
            final_image = np.squeeze(final_image)

        return (final_image * 255).astype(np.uint8)

    def depth_to_rgb(
        self,
        depth_map: np.ndarray,
        cmap: str = 'plasma'
    ) -> np.ndarray:
        """
        Convert a 2D depth map to a 3-channel RGB image using a colormap.
        """
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            raise ValueError("Input must be a 2D numpy array "
                             "(grayscale depth map)")

        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        if max_val - min_val == 0:
            norm_depth = np.zeros_like(depth_map, dtype=np.float32)
        else:
            norm_depth = (depth_map - min_val) / (max_val - min_val)

        colormap = plt.get_cmap(cmap)
        colored = colormap(norm_depth)  # Returns RGBA
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        return rgb
