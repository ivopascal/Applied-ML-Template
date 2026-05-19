import numpy as np

from features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import PATCH_SIZE


class RGBNormalizationPreprocessor(BasePreprocessor):
    """Validate shape and convert an RGB image patch to float32.
    This is the single entry point for all four views. It enforces
    the shape contract, ensures a consistent dtype, and handles
    corrupted (non-finite) pixel values.
    """

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Validate and normalize a raw RGB image patch.

        Args:
            image_patch: Input image of expected shape
                ``(PATCH_SIZE, PATCH_SIZE, 3)``. Accepts uint8,
                float32, or float64 dtypes.

        Raises:
            ValueError: If ``image_patch`` does not have the expected
                shape ``(PATCH_SIZE, PATCH_SIZE, 3)``.

        Returns:
            Validated image as a float32 array of shape
            ``(PATCH_SIZE, PATCH_SIZE, 3)`` with all values finite.
            Non-finite values are replaced with ``0.0``.
        """
        image = np.asarray(image_patch, dtype=np.float32)

        if image.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            raise ValueError(
                f"Expected shape ({PATCH_SIZE}, {PATCH_SIZE}, 3), "
                f"got {image.shape}."
            )
        if not np.isfinite(image.sum()):  # NaN or inf will lead to undefined sum
            bad_mask = ~np.isfinite(image)  # bitwise NOT operator
            image = np.where(bad_mask, np.float32(0.0), image)

        return image
