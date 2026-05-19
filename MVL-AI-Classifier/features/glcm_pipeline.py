import numpy as np

from features.base_processor import BasePreprocessor
from features.rgb_normalization_pipeline import RGBNormalizationPreprocessor
from features.rgb_gray_pipeline import RGBToGrayPreprocessor


class GLCMPreprocessor(BasePreprocessor):
    """
    Compute normalized Gray-Level Co-occurrence Matrices (GLCMs) from a
    256x256 RGB image patch.

    Four spatial offsets are computed:
        horizontal    (dx=+1, dy= 0)
        vertical      (dx= 0, dy=+1)
        diagonal      (dx=+1, dy=+1)
        anti-diagonal (dx=-1, dy=+1)

    Output
    ------
    np.ndarray of shape (4, n_levels, n_levels), dtype float32.
        Normalized co-occurrence probabilities.
        axis 0 order: [horizontal, vertical, diagonal, anti_diagonal]
    """

    _OFFSETS = [
        (1, 0),  # horizontal
        (0, 1),  # vertical
        (1, 1),  # diagonal
        (-1, 1),  # anti-diagonal
    ]

    def __init__(
        self,
        n_levels: int = 32,
        symmetric: bool = True,
    ):
        """
        Parameters
        ----------
        n_levels : int
            Number of quantization levels for intensity space, default 32.
        symmetric : bool
            If True, GLCM is symmetrized: G = G + G.T.
        """
        if not isinstance(n_levels, (int, np.integer)) or n_levels <= 0:
            raise ValueError("n_levels must be a positive integer.")

        # 256 possible levels of brightness per pixel, reduced to n_levels for robustness
        self.n_levels = int(n_levels)
        self.symmetric = symmetric
        self._scale = np.float32(self.n_levels / 255.0)

        self._normalization = RGBNormalizationPreprocessor()
        self._to_gray = RGBToGrayPreprocessor()

    def _compute_glcm(
        self,
        qimg: np.ndarray,
        dx: int,
        dy: int,
    ) -> np.ndarray:
        """
        Compute one normalized GLCM for spatial offset (dx, dy).

        Returns
        -------
        np.ndarray of shape (n_levels, n_levels), dtype float32.
        """
        h, w = qimg.shape  # quantized image shape

        # determine valid pixel pairs
        y_ref = slice(max(0, -dy), h - max(0, dy))
        y_nb = slice(max(0, dy), h - max(0, -dy))
        x_ref = slice(max(0, -dx), w - max(0, dx))
        x_nb = slice(max(0, dx), w - max(0, -dx))

        # create two aligned vectors for reference and neighbor pixels
        # ravel flattens arrays into vectors
        ref = qimg[y_ref, x_ref].ravel()
        nb = qimg[y_nb, x_nb].ravel()

        # count matrix with row = reference intensity, column = neighbor intensity
        G = np.zeros((self.n_levels, self.n_levels), dtype=np.float32)
        np.add.at(G, (ref, nb), np.float32(1.0))

        if self.symmetric:  # (3,7) == (7,3)
            G = G + G.T

        # normalization
        total = G.sum()
        if total > 0:
            G /= total

        return G

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        image = self._normalization(image_patch)
        gray = self._to_gray(image)
        gray_q = np.floor(gray * self._scale).astype(np.int32)
        gray_q = np.clip(gray_q, 0, self.n_levels - 1)

        glcms = np.stack(
            [self._compute_glcm(gray_q, dx, dy) for dx, dy in self._OFFSETS],
            axis=0,
        )  # (4, n_levels, n_levels), float32

        return glcms
