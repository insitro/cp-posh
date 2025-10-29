import io
import os
import pickle
from typing import Any, Tuple, Union, cast
import numpy as np
import tifffile as tif
import boto3
from botocore.config import Config


def _get_bucket_key(
    path: Union[str, None] = None,
    bucket: Union[str, None] = None,
    key: Union[str, None] = None,
) -> Tuple[str, str]:
    if path is not None:
        if bucket is not None or key is not None:
            raise ValueError("Either path is None or bucket and key are both None.")
        elif not path.startswith("s3://"):
            raise ValueError("Input path must start with s3://")

        bucket, key = path.replace("s3://", "", 1).split("/", 1)

    else:
        if bucket is None and key is None:
            raise ValueError("All inputs cannot be None.")

        bucket = cast(str, bucket)
        key = cast(str, key)
    return bucket, key


def get_well_loc(row: int, column: int) -> str:
    """
    Get well location corresponding to row and column

    Parameters
    ----------
    row : int
        well row
    column : int
        well column

    Returns
    -------
    str
        well_loc
    """
    well_loc = chr(ord("A") + row - 1) + str(f"{column:02}")
    return well_loc


def read_image(
    path: str,
    image_type="hwc",
    loaded_type="chw",
) -> np.ndarray:
    """
    Read image (tiff/numpy image formats)

    Parameters
    ----------
    path : str
        Image path (local / s3 location - of numpy image)
    image_type : ImageStorageFormat
        ordering of channels in the image file specified by ImageStorageFormat,
        by default ImageStorageFormat.HWC
    loaded_type : ImageLoadedFormat
        Specifies what the loaded result should be. Defaults to the standard pyxcell choice.

    Returns
    -------
    np.ndarray
        A 2D or 3D tensor. The ordering is specified by the inputs.
    """
    s3_client = boto3.client(
        "s3",
        config=Config(retries=dict(max_attempts=10)),
    )

    bucket, key = _get_bucket_key((path))
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
    if key.endswith(".npy"):
        # read numpy inputs
        img = np.load(io.BytesIO(s3_obj["Body"].read()), allow_pickle=True)
    elif path.endswith(".tif") or path.endswith(".tiff"):
        _tif_ = tif.TiffFile(io.BytesIO(s3_obj["Body"].read()))
        if image_type == "hwc":
            img = _tif_.asarray(key=0)
        else:
            img = _tif_.asarray()
    else:
        raise ValueError(f"Unsupported file format {key}")

    if image_type != loaded_type:
        if image_type == "hwc" and loaded_type == "chw":
            img = np.moveaxis(img, -1, 0)
        elif image_type == "chw" and loaded_type == "hwc":
            img = np.moveaxis(img, 0, -1)
    return img


def load_data(path: str) -> Any:
    """
    Helper function to load pkl and numpy object files

    Parameters
    ----------
    path : str
        path to data

    Returns
    -------
    Any
    """
    blob = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            if path.endswith(".pkl"):
                blob = pickle.load(f)
            elif path.endswith(".npy"):
                blob = np.load(f)
    return blob
