class RGBNormalizationPreprocessor(BasePreprocessor):
    """
    Preprocess RGB image patches by:
      - validate RGB shape,
      - convert dtype,
      - normalize integer-valued images,
      - remove NaN/Inf values.
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, image_patch):
        image = np.asarray(image_patch)

        # Validate JPEG patch dimensions.
        if image.ndim != 3 or image.shape != (256, 256, 3):
            raise ValueError(
                f"Expected RGB image shape (256, 256, 3), got {image.shape}."
            )

        # Convert to floating point.
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float64)
        else:
            image = image.astype(np.float64, copy=False)

        # Remove invalid value
        if not np.isfinite(image).all():
            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return image
