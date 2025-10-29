import enum
import numpy as np
import decorator
import pandas as pd
import skimage
from scipy.spatial import cKDTree

class AlignMethod(enum.Enum):
    DAPI = 0
    SBS_MEAN = 1


@decorator.decorator
def applyIJ(f, arr, *args, **kwargs):
    """
    Apply a function that expects 2D input to the trailing two
    dimensions of an array. The function must output an array whose shape
    depends only on the input shape.
    # code adapted from https://github.com/feldman4/OpticalPooledScreens/blob/master/ops/process.py
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    # kwargs are not actually getting passed in?
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


class Align:
    """
    functions for alignment/registration between cycles used by OPS sequencing pipeline
    # code adapted from https://github.com/feldman4/OpticalPooledScreens/blob/master/ops/process.py
    """

    @staticmethod
    def normalize_by_percentile(data_: np.ndarray, q_norm: int = 70):
        """
        normalize image by q_norm percentile

        Parameters
        ----------
        data_ : np.ndarray
            input image
        q_norm : int, optional
            q_norm used for percentile scaling, by default 70

        Returns
        -------
        np.ndarray
            normalized image
        """
        shape = data_.shape
        shape = shape[:-2] + (-1,)
        p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
        normed = data_ / p
        return normed

    @staticmethod
    @applyIJ
    def filter_percentiles(data, q1, q2):
        """
        Replaces data outside of percentile range [q1, q2]
        with uniform noise over the range [q1, q2]. Useful for
        eliminating alignment artifacts due to bright features or
        regions of zeros.
        """
        x1, x2 = np.percentile(data, [q1, q2])
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    @applyIJ
    def filter_values(data, x1, x2):
        """
        Replaces data outside of value range [x1, x2]
        with uniform noise over the range [x1, x2]. Useful for
        eliminating alignment artifacts due to bright features or
        regions of zeros.
        """
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    def fill_noise(data, mask, x1, x2):
        filtered = data.copy()
        rs = np.random.RandomState(0)
        filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
        return filtered

    @staticmethod
    def calculate_offsets(data_, upsample_factor):
        target = data_[0]
        offsets = []
        for i, src in enumerate(data_):
            if i == 0:
                offsets += [(0, 0)]
            else:
                offset, _, _ = skimage.registration.phase_cross_correlation(
                    src, target, upsample_factor=upsample_factor
                )
                offsets += [offset]
        return np.array(offsets)

    @staticmethod
    def apply_offsets(data_, offsets):
        warped = []
        for frame, offset in zip(data_, offsets):
            if offset[0] == 0 and offset[1] == 0:
                warped += [frame]
            else:
                # skimage has a weird (i,j) <=> (x,y) convention
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_ = skimage.transform.warp(frame, st, preserve_range=True)
                warped += [frame_.astype(data_.dtype)]

        return np.array(warped)

    @staticmethod
    def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
        filtered = Align.filter_percentiles(
            Align.apply_window(data_, window), q1=q1, q2=q2
        )
        offsets = Align.calculate_offsets(filtered, upsample_factor=upsample_factor)
        return Align.apply_offsets(data_, offsets)

    @staticmethod
    def align_between_cycles(
        data, channel_index, upsample_factor=4, window=1, return_offsets=False
    ):
        # offsets from target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor)

        # apply to all channels
        warped = []
        for data_ in data.transpose([1, 0, 2, 3]):
            warped += [Align.apply_offsets(data_, offsets)]

        aligned = np.array(warped).transpose([1, 0, 2, 3])
        if return_offsets:
            return aligned, offsets
        else:
            return aligned

    @staticmethod
    def apply_window(data, window):
        height, width = data.shape[-2:]

        def find_border(x):
            return int((x / 2.0) * (1 - 1 / float(window)))

        i, j = find_border(height), find_border(width)
        return data[..., i : height - i, j : width - j]


def align_SBS(
    image: np.ndarray,
    dapi_index: int,
    method: AlignMethod = AlignMethod.DAPI,
    upsample_factor: float = 1,
    window: int = 2,
    cutoff: int = 1,
    align_within_cycle: bool = True,
):
    """
    Rigid alignment of sequencing cycles and channels.
    Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
    A centered subset of data is used if `window` is greater
    than one. Subpixel alignment is done if `upsample_factor` is greater than
    one (can be slow).
    # code adapted from https://github.com/feldman4/OpticalPooledScreens/blob/master/ops/process.py

    Parameters
    ----------
    image : np.ndarray
        input sequencing image of shape (n_cycles, n_channels, h, w)
    dapi_index : int
        index of dapi channel in the sequencing image stack
    method : str, optional
        method to use for alignment, by default
    upsample_factor : float, optional
        upsample factor for aligning between cycles, by default 2
    window : int, optional
        window of alignment, by default 1
    cutoff : int, optional
        sequencing base threshold for alignment, by default 1
    align_within_cycle : bool, optional
        flag to indicate alignment within cycle, by default True

    Returns
    -------
    np.ndarray
        aligned image stack
    """

    # Input image should be of shape (n_cycles, n_channels, h, w)
    assert len(image.shape) == 4

    # align between SBS channels for each cycle
    aligned = image.copy()
    if align_within_cycle:

        def align_it(x):
            return Align.align_within_cycle(
                x, window=window, upsample_factor=upsample_factor
            )

        aligned[:, :] = np.array([align_it(x) for x in aligned[:, :]])

    if method == AlignMethod.DAPI:
        # align cycles using the DAPI channel
        aligned = Align.align_between_cycles(
            aligned,
            channel_index=dapi_index,
            window=window,
            upsample_factor=upsample_factor,
        )
    elif method == AlignMethod.SBS_MEAN:
        # calculate cycle offsets using the average of SBS channels
        target = Align.apply_window(
            aligned[:, [ch for ch in range(0, aligned.shape[1]) if ch != dapi_index]],
            window=window,
        ).max(axis=1)
        normed = Align.normalize_by_percentile(target)
        normed[normed > cutoff] = cutoff
        offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
        # apply cycle offsets to each channel
        for channel in range(aligned.shape[1]):
            aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

    return aligned


def deduplicated_barcode_locations(
    barcodes_df: pd.DataFrame, deduplicate_distance: float
) -> pd.DataFrame:
    """
    Deduplicate barcode locations by removing duplicates within distance threshold

    Parameters
    ----------
    barcodes_df : pd.DataFrame
        dataframe of barcode locations with columns ['x', 'y', 'z', 'target']
    deduplicate_distance : float
        distance threshold for deduplication

    Returns
    -------
    pd.DataFrame
        deduplicated barcode locations
    """
    barcodes_df = barcodes_df.reset_index()
    tree = cKDTree(barcodes_df[["x", "y"]].values)

    # find pairs of points within the deduplicate_distance
    pairs = tree.query_pairs(deduplicate_distance)

    # create a set to keep track of points to remove
    to_remove = set()
    for i, j in pairs:
        to_remove.add(j)

    # Filter out the points to remove
    deduplicated_df = barcodes_df.drop(to_remove)

    return deduplicated_df
