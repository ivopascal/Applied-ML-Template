import numpy as np

from features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import DEFAULT_EPSILON, PATCH_SIZE
from features.rgb_normalization_pipeline import RGBNormalizationPreprocessor
from features.rgb_gray_pipeline import RGBToGrayPreprocessor


class AzimuthalPowerSpectrumPreprocessor(BasePreprocessor):
    """
    Compute a mean-based azimuthal (radial) power spectrum from an RGB patch.
    Output
    ------
    np.ndarray of shape (n_bins,), dtype float32.
        Values are log-transformed mean power per radial frequency bin.
    """

    def __init__(
        self,
        n_bins: int = 64,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """
        Parameters
        ----------
        n_bins : int
            Number of radial frequency bins.
            Recommended: 64 for 256x256 patches.
        epsilon : float
            Numerical stability constant for log transform.
        """
        if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")

        self.n_bins = int(n_bins)
        self.epsilon = np.float32(epsilon)

        self._normalization = RGBNormalizationPreprocessor()
        self._to_gray = RGBToGrayPreprocessor()

        # Precompute Hann window for better edge handling, reducing boundary artifacts
        w = np.hanning(PATCH_SIZE).astype(np.float32)
        self._window_2d = np.outer(w, w)

        # Precompute bin assignments and per-bin pixel counts.
        self._bin_idx, self._bin_counts = self._build_bin_index()

    def _build_bin_index(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute radial bin assignment for each FFT coefficient and
        the number of coefficients per bin. Basically a lookup table for the
        frequency pixels per radial bin

        Returns
        -------
        bin_idx : np.ndarray of shape (PATCH_SIZE * PATCH_SIZE,), int32.
        bin_counts : np.ndarray of shape (n_bins,), float32.
            Per-bin coefficient count. Used as denominator for mean.
            Precomputed because it is constant across all images.
        """
        cx = cy = PATCH_SIZE // 2  # reference point
        # map coordinates
        y, x = np.meshgrid(
            np.arange(PATCH_SIZE),
            np.arange(PATCH_SIZE),
            indexing="ij",
        )
        # radial distance from the center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        r_max = float(r.max())

        # create radial bins
        bin_edges = np.linspace(0.0, r_max, self.n_bins + 1, dtype=np.float32)

        # assign FFT pixels to a bin + edge case handling
        bin_idx = np.digitize(r.ravel(), bin_edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1).astype(np.int32)

        # count number of coefficients
        bin_counts = np.bincount(bin_idx, minlength=self.n_bins).astype(np.float32)

        return bin_idx, bin_counts

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        image = self._normalization(image_patch)
        gray = self._to_gray(image)

        # Remove DC component and apply Hann window
        gray_windowed = (gray - gray.mean()) * self._window_2d

        # 1D power spectrum
        F_shifted = np.fft.fftshift(np.fft.fft2(gray_windowed))
        power_flat = (np.abs(F_shifted) ** 2).astype(np.float32).ravel()

        # Mean power per radial bin
        sums = np.bincount(
            self._bin_idx,
            weights=power_flat,
            minlength=self.n_bins,
        ).astype(np.float32)

        # bins with zero counts stay at zero.
        spectrum = np.where(
            self._bin_counts > 0,
            sums / self._bin_counts,
            np.float32(0.0),
        )

        return np.log(spectrum + self.epsilon).astype(np.float32)
