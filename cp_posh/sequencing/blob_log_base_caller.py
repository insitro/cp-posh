import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import feature, morphology, util
from scipy import ndimage as ndi
from cp_posh.utils import scaling


def _log_ndi(data, sigma=1, *args, **kwargs):
    """
    Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J).
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    Inverts output and converts back to uint16.
    """
    arr_ = -1 * ndi.gaussian_laplace(data.astype(float), sigma, *args, **kwargs)
    arr_ = np.clip(arr_, 0, 65535) / 65535

    # skimage precision warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return util.img_as_uint(arr_)


def _max_filter(data, width):
    """
    Apply a maximum filter in a window of `width`.
    """
    maxed = ndimage.maximum_filter(data, size=(width, width))
    return maxed


def call_bases(
    sbs_image: np.ndarray,
    base_map: Dict[int, str],
    kernel_size: int = 5,
    max_filter_kernel_size: int = 3,
    blob_detector_threshold: float = 0.1,
    lapog_filter_image: bool = True,
) -> pd.DataFrame:
    """
    call bases using a Laplacian of Gaussian blob detection method

    Parameters
    ----------
    sbs_image : np.ndarray
        input sequencing by synthesis image
    base_map : Dict[int, str]
        mapping of base position to base name
    kernel_size : int, optional
        kernel size for white_tophat filtering, by default 5
    max_filter_kernel_size : int, optional
        kernel size for max filter, by default 3
    blob_detector_threshold : float, optional
        scale space maxima threshold for blob detector, by default 0.1
    lapog_filter_image: bool, optional
        filter image by a laplacian of gaussian filter to pick sharp peaks
        by default, True

    Returns
    -------
    pd.DataFrame
        dataframe of detected spots
    """

    assert kernel_size % 2 == 1
    assert max_filter_kernel_size % 2 == 1

    # Step 1: Log-transform image and min-max rescale intensity
    spots_image = util.img_as_uint(
        scaling.rescale_intensity(np.log(sbs_image.astype(np.float32) + 1), 0, 100)
    )

    # Step 2: Obtain max image per pixel across 4 channels (4 bp)
    spots_image = np.max(spots_image, axis=0)

    # Step 3: Filter auto-fluorescence background using White-Tophat filter
    spots_image = morphology.white_tophat(
        spots_image, footprint=morphology.disk(kernel_size)
    )

    # Step 4: Apply Max-filter and min-max rescale intensity (convert to uint8)
    spots_image = util.img_as_uint(
        scaling.rescale_intensity(
            _max_filter(spots_image, width=max_filter_kernel_size), 0, 100
        )
    )

    # Step 5: Laplacian-of-Gaussian blob detection
    spots = feature.blob_log(
        spots_image,
        min_sigma=1,
        max_sigma=kernel_size,
        num_sigma=kernel_size * 2,
        threshold=blob_detector_threshold,
        exclude_border=kernel_size,
    )

    if lapog_filter_image:
        sbs_intensity_image = _log_ndi(sbs_image)

    base_index = {v: k for k, v in base_map.items()}

    base_calls = []
    for spot in spots:
        y, x, r = spot.astype(int)
        spot_image = sbs_intensity_image[
            :,
            max(0, y - kernel_size // 2) : min(
                y + kernel_size // 2 + 1, sbs_image.shape[1]
            ),
            max(0, x - kernel_size // 2) : min(
                x + kernel_size // 2 + 1, sbs_image.shape[2]
            ),
        ]
        spot_intensities = np.max(spot_image, axis=(1, 2))
        base = base_map[int(np.argmax(spot_intensities)) + 1]

        base_calls.append(
            {
                "base": base,
                "x": x,
                "y": y,
                "spot_size": r,
                "i_C": spot_intensities[base_index["C"] - 1],
                "i_A": spot_intensities[base_index["A"] - 1],
                "i_T": spot_intensities[base_index["T"] - 1],
                "i_G": spot_intensities[base_index["G"] - 1],
            }
        )
    return pd.DataFrame(base_calls)
