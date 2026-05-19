import numpy as np

from features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import DEFAULT_EPSILON, PATCH_SIZE
from features.rgb_normalization_pipeline import RGBNormalizationPreprocessor
from features.rgb_gray_pipeline import RGBToGrayPreprocessor


class AzimuthalPowerSpectrumPreprocessor(BasePreprocessor):
    """Compute mean-based azimuthal (radial) power spectrum from an RGB patch.

    Pipeline:
        RGB → validate/float32 → grayscale → DC removal → Hann window
        → FFT2 → power spectrum → radial binning (mean) → log transform

    The 2D power spectrum is reduced to a 1D radial profile by averaging
    power within concentric frequency rings. This produces a compact,
    rotationally-invariant summary of which spatial frequencies are
    present in the image.
    """
    

    def __init__(
        self,
        n_bins: int = 64,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize the APS preprocessor with precomputed lookup tables.

        Args:
            n_bins: Number of radial frequency bins in the output spectrum.
                Recommended 64 for 256x256 patches, giving bin width ≈2.8
                pixels in frequency space.
            epsilon: Small constant added inside the log transform to
                prevent ``log(0)``.
        """
        
        self.n_bins = int(n_bins)
        self.epsilon = np.float32(epsilon)

        self._normalization = RGBNormalizationPreprocessor()
        self._to_gray = RGBToGrayPreprocessor()

        # Precompute 2D Hann window to taper image edges to zero,
        # preventing spectral leakage from discontinuities at the image boundary
        w = np.hanning(PATCH_SIZE).astype(np.float32)
        self._window_2d = np.outer(w, w)
        # Shape: (PATCH_SIZE, PATCH_SIZE), float32
        # Center values ≈ 1.0 (signal preserved), edge values = 0.0 (signal tapered to zero)

        # Precompute bin assignments and per-bin pixel counts
        # These depend only on the grid geometry, not on the image content.
        self._bin_idx, self._bin_counts = self._build_bin_index()

    def _build_bin_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a lookup table mapping each FFT grid position to a radial bin.

        The 256x256 FFT grid (after fftshift) has its DC component at
        position (128, 128). Each grid position has a radial distance
        from this center, which determines its frequency bin assignment.

        This mapping is purely geometric, as it depends only on the grid
        size and number of bins, not on any image data.

        Returns:
            Tuple of:
                - ``radial_bin_indices``: int32 array of shape
                  ``(PATCH_SIZE * PATCH_SIZE,)`` where element ``i``
                  is the bin index for flattened grid position ``i``.
                - ``coefficients_per_bin``: float32 array of shape
                  ``(n_bins,)`` counting how many FFT grid positions
                  fall in each radial bin. Used as denominator for
                  mean computation.
        """
        # Center of the shifted FFT grid (DC component location).
        center_x = center_y = PATCH_SIZE // 2 # 128 for PATCH_SIZE=256

        # Coordinate grids for all 256×256 positions
        # indexing="ij" results in a matrix indexing, with
        #   row_coords[i, j] = i,  col_coords[i, j] = j
        row_coords, col_coords = np.meshgrid(
            np.arange(PATCH_SIZE),
            np.arange(PATCH_SIZE),
            indexing="ij",
        )
        # Euclidean distance from each grid position to center.
        radius = np.sqrt((col_coords - center_x) ** 2 + (row_coords - center_y) ** 2).astype(np.float32)
        radius_max = float(radius.max())

        # Create n_bins equally-spaced radial bins from 0 to max_radius.
        # n_bins + 1 edges define n_bins intervals:
        #   bin 0: [edge_0, edge_1), bin 1: [edge_1, edge_2), ...
        bin_edges = np.linspace(0.0, radius_max, self.n_bins + 1, dtype=np.float32)

        # Assign each grid position to a bin based on its radial distance.
        # np.digitize returns 1-based indices; subtract 1 for 0-based.
        # right=False means intervals are [left, right) — closed-left, open-right.
        bin_idx = np.digitize(radius.ravel(), bin_edges, right=False) - 1
        
        # Clip to valid range [0, n_bins-1].
        # Necessary because positions exactly at max_radius get index n_bins
        # from digitize (they exceed the last left edge)
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1).astype(np.int32)

        # Count grid positions per bin (constant across all images).
        # Inner bins (small radius) contain fewer positions because the
        # ring area 2πr·dr is smaller, outer bins contain more.
        bin_counts = np.bincount(bin_idx, minlength=self.n_bins).astype(np.float32)

        return bin_idx, bin_counts

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Compute the log-transformed radial power spectrum.

        Args:
            image_patch: RGB image of shape ``(PATCH_SIZE, PATCH_SIZE, 3)``
            
        Returns:
            Log-transformed mean power per radial frequency bin,
            shape ``(n_bins,)`` as float32.
        """
        image = self._normalization(image_patch)
        gray = self._to_gray(image)

        # DC removal prevents the zero-frequency spike from
        # dominating all radial bins after averaging. Apply Hann window
        # (256, 256)
        gray_windowed = (gray - gray.mean()) * self._window_2d

        #fft2: spatial domain → frequency domain (complex coefficients).
        # fftshift: move DC from corner (0,0) to center (128,128) to match
        #   the radial bin lookup table built in _build_radial_bin_lookup.
        # shape (256, 256)
        F_shifted = np.fft.fftshift(np.fft.fft2(gray_windowed))
        
        # Power = |F|², discarding phase (position/shift of each frequency
        # component) and keeping only energy (how strong each frequency is).
        # shape (65536,)
        power_flat = (np.abs(F_shifted) ** 2).astype(np.float32).ravel()

        # Sum power values within each radial bin using the
        # precomputed bin assignment lookup table.
        sums = np.bincount(
            self._bin_idx,
            weights=power_flat,
            minlength=self.n_bins,
        ).astype(np.float32)

        # bins with zero counts stay at zero, shape (n_bins,)
        spectrum = np.where(
            self._bin_counts > 0,
            sums / self._bin_counts,
            np.float32(0.0),
        )
        
        # Power values span 10+ orders of magnitude; log maps this to
        # approximately [-20, 20], which neural networks handle better.
        # epsilon prevents log(0) for empty bins.
        
        return np.log(spectrum + self.epsilon).astype(np.float32)
