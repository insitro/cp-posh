# adapted from Starfish codebase: https://github.com/spacetx/starfish/blob/master/notebooks/ISS.ipynb
from typing import List
import numpy as np
import pandas as pd
from starfish import Codebook, ImageStack
from starfish.image import Filter
from starfish.spots import DecodeSpots, FindSpots
from starfish.types import Axes, Features, FunctionSource

from cp_posh.sequencing.baselines import utils


def load_codebook(codebook_filename: str, barcode_indices: List[int]) -> Codebook:
    """
    Load the codebook from parquet file

    Parameters
    ----------
    codebook_filename : str
        path to the codebook file
    barcode_indices : List[int]
        list of indices to extract from the barcode

    Returns
    -------
    Codebook
        codebook object
    """
    library = pd.read_parquet(codebook_filename)
    library["barcode"] = library.sgRNA.apply(
        lambda x: "".join([x[i] for i in barcode_indices])
    )
    base_map = {"C": 0, "A": 1, "T": 2, "G": 3}
    codebook = []
    for row in library.itertuples():
        codeword = []
        for idx, base in enumerate(row.barcode):
            codeword.append(
                {
                    Axes.ROUND.value: idx,
                    Axes.CH.value: base_map[base],
                    Features.CODE_VALUE: 1,
                }
            )
        codebook.append({Features.CODEWORD: codeword, Features.TARGET: row.gene_id})
    codebook = Codebook.from_code_array(codebook)
    return codebook


def call_barcodes(
    image_array: np.ndarray, codebook_filename: str, barcode_indices: List[int]
) -> pd.DataFrame:
    """
    Call barcodes from the image array with codebook using starfish pipeline

    Parameters
    ----------
    image_array : np.ndarray
        input image array of shape (R, C, H, W)
        where R is the number of rounds, C is the number of channels,
        H is the height, and W is the width
    codebook_filename : str
        path to the codebook parquet file
    barcode_indices : List[int]
        list of indices to extract from the barcode

    Returns
    -------
    pd.DataFrame
        dataframe of barcode locations with columns ['x', 'y', 'z', 'target']
    """
    # align the image across SBS cycles
    aligned_image = utils.align_SBS(
        image_array, dapi_index=4, method=utils.AlignMethod.DAPI
    )

    # load barcode codebook
    codebook = load_codebook(
        codebook_filename=codebook_filename, barcode_indices=barcode_indices
    )

    # remove the last channel (DAPI), add z-axis and convert aligned images into starfish ImageStack
    aligned_sbs = aligned_image[barcode_indices, 0:4, :, :][:, :, np.newaxis, :, :]
    image_stack = ImageStack.from_numpy(aligned_sbs)

    # mask and filter the image using parameters from the starfish pipeline
    masking_radius = 15
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)
    filtered_imgs = filt.run(image_stack, verbose=True, in_place=False)

    # find barcode spots
    bd = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type="mean",
    )

    dots_max = image_stack.reduce(
        (Axes.ROUND, Axes.CH, Axes.ZPLANE), func=FunctionSource.np("max")
    )
    spots = bd.run(image_stack=filtered_imgs, reference_image=dots_max)

    decoder = DecodeSpots.PerRoundMaxChannel(codebook=codebook)
    decoded_intensities = decoder.run(spots=spots)

    # extract barcode locations
    spots_df = decoded_intensities.to_features_dataframe()

    # select relevant columns
    barcode_locations = spots_df[["x", "y", "z", "target"]]
    return barcode_locations
