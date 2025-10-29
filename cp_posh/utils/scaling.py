import numpy as np


def rescale_intensity(
    image: np.ndarray,
    min_percentile: float = 0.1,
    max_percentile: float = 99.9,
    clip: bool = False,
) -> np.ndarray:
    """
    Rescale intensity of input image by min and max percentile intensities

    Parameters
    ----------
    image : np.ndarray
        input image of shape (C, H, W) or (H, W)
    min_percentile : float, optional
        min percentile, by default 0.1
    max_percentile : float, optional
        max percentile, by default 99.9
    clip: bool, optional
        flag to clip the intensities above or below the percentile values,
        by default False

    Returns
    -------
    np.ndarray
        re-scaled image
    """

    if len(image.shape) == 2:
        low = np.percentile(image, min_percentile, axis=(0, 1))
        high = np.percentile(image, max_percentile, axis=(0, 1))

        # Cannot divide by zero
        if low == high:
            low = 0
            high = max(high, 1)
    else:
        n_channels = image.shape[0]
        low = np.percentile(image, min_percentile, axis=(1, 2)).reshape(
            n_channels, 1, 1
        )
        high = np.percentile(image, max_percentile, axis=(1, 2)).reshape(
            n_channels, 1, 1
        )

        # Cannot divide by zero
        bad_idxs = np.where(high == low)
        high[bad_idxs] = np.max(high[bad_idxs], initial=1)
        low[bad_idxs] = 0

    image = (image - low) / (high - low)

    if clip:
        image = np.clip(image, 0, 1)

    return image
