import numpy as np

from features.base_processor import BasePreprocessor

class RGBToGrayPreprocessor(BasePreprocessor):
    """Convert a float32 RGB image to single-channel grayscale.

    Uses the ITU-R BT.601 luminance formula:

        ``gray = 0.299 * R + 0.587 * G + 0.114 * B``

    This class performs ONLY color-space conversion. No normalization,
    rescaling, or quantization is applied. The output range mirrors
    the input range exactly.
    """

    _COEFFICIENTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Convert an RGB image to grayscale via luminance projection.

        Args:
            image: Float32 RGB image of shape ``(H, W, 3)``.

        Raises:
            ValueError: If ``image`` is not 3-dimensional or does not
                have exactly 3 channels along the last axis.

        Returns:
            Grayscale image of shape ``(H, W)`` as float32.
            Value range mirrors input (e.g. [0, 255] for uint8 sources
            that were converted to float32 by RGBNormalizationPreprocessor).
        """
        if image_patch.ndim != 3 or image_patch.shape[2] != 3:
            raise ValueError(f"Expected shape (H, W, 3), got {image_patch.shape}.")

        # Matrix-vector dot product along the channel axis.
        # image shape: (H, W, 3) @ (3,) -> (H, W)

        gray = image_patch.astype(np.float32, copy=False) @ self._COEFFICIENTS

        return gray
