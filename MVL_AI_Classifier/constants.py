# Side length of the square input patch in pixels.
# Must be a power of 2 (efficient FFT) and divisible by JPEG_BLOCK_SIZE.
PATCH_SIZE = 256

# JPEG standard mandates 8x8 pixel blocks for DCT.
JPEG_BLOCK_SIZE = 8

# Number of non-overlapping JPEG blocks along each spatial dimension.
# 256 / 8 = 32 blocks per row and 32 blocks per column.
JPEG_BLOCKS_PER_DIM = PATCH_SIZE // JPEG_BLOCK_SIZE

# JPEG level shift: maps unsigned pixel values [0, 255] to
# signed values [-128, 127] for zero-centered DCT computation.
JPEG_RECENTER_VALUE = 128.0

# Small constant added to denominators and log arguments to prevent
# division-by-zero and log(0)
DEFAULT_EPSILON = 1e-8