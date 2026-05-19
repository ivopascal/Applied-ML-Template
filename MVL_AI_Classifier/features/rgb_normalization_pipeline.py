import numpy as np

from features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import PATCH_SIZE


class RGBNormalizationPreprocessor(BasePreprocessor):
    """
    Validate and normalize an RGB image patch to float32

    Output
    ------
    np.ndarray of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype float32.
    """

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
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
