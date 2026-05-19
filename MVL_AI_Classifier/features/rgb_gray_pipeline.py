import numpy as np

from features.base_processor import BasePreprocessor


class RGBToGrayPreprocessor(BasePreprocessor):
    """
    Convert a float32 RGB image to single-channel grayscale.

    Uses the ITU-R BT.601 luminance formula:

        gray = 0.299 * R + 0.587 * G + 0.114 * B

    Output
    ------
    np.ndarray of shape (H, W), dtype float32.
    """

    _COEFFICIENTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        if image_patch.ndim != 3 or image_patch.shape[2] != 3:
            raise ValueError(f"Expected shape (H, W, 3), got {image_patch.shape}.")

        # Matrix-vector dot product along the channel axis.
        # image shape: (H, W, 3) @ (3,) -> (H, W)

        gray = image_patch.astype(np.float32, copy=False) @ self._COEFFICIENTS

        return gray
