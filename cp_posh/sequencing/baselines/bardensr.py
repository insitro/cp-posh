# adapted from https://github.com/jacksonloper/bardensr/blob/master/examples/basics.ipynb
from typing import List
import numpy as np
import pandas as pd

# follow instructions in https://github.com/jacksonloper/bardensr/tree/master/ to install bardensr
# bardensr is not part of cp-posh dependencies
import bardensr
from cp_posh.sequencing.baselines import utils


def load_codebook(codebook_filename: str, barcode_indices: List[int]) -> np.ndarray:
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
    np.ndarray
        codebook array
    """
    library = pd.read_parquet(codebook_filename)
    library["barcode"] = library.sgRNA.apply(
        lambda x: "".join([x[i] for i in barcode_indices])
    )
    base_map = {
        "C": [1, 0, 0, 0],
        "A": [0, 1, 0, 0],
        "T": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }
    codebook = []
    for row in library.itertuples():
        codeword = []
        for _, base in enumerate(row.barcode):
            codeword.append(base_map[base])
        codebook.append(codeword)
    return np.array(codebook).transpose(1, 2, 0)


def call_barcodes(
    image_array: np.ndarray, codebook_filename: str, barcode_indices: List[int]
) -> pd.DataFrame:
    """
    Call barcodes from the image array with codebook using bardensr pipeline

    Parameters
    ----------
    image_array : np.ndarray
        input image array of shape (R, C, H, W)
    codebook_filename : str
        path to the codebook parquet file
    barcode_indices : List[int]
        list of indices to extract from the barcode

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # align the image across SBS cycles
    aligned_image = utils.align_SBS(
        image_array, dapi_index=4, method=utils.AlignMethod.DAPI
    )

    # remove the last channel (DAPI) and add z-axis
    aligned_sbs = aligned_image[barcode_indices, 0:4, :, :][:, :, np.newaxis, :, :]

    # load codebook from parquet file
    codebook = load_codebook(codebook_filename, barcode_indices)
    library = pd.read_parquet(codebook_filename)

    # run bardensr pipeline
    R, C, J = codebook.shape
    F = R * C
    Xflat = aligned_sbs.reshape((R * C,) + aligned_sbs.shape[-3:])

    codeflat = codebook.reshape((F, -1))
    Xnorm = bardensr.preprocessing.minmax(Xflat)
    Xnorm = bardensr.preprocessing.background_subtraction(Xnorm, [0, 10, 10])
    Xnorm = bardensr.preprocessing.minmax(Xnorm)

    # estimate density for evidence tensor
    evidence_tensor = bardensr.spot_calling.estimate_density_singleshot(
        Xnorm, codeflat, noisefloor=0.05
    )

    # find peaks in evidence tensor
    thresh = 0.72
    barcode_locations = bardensr.spot_calling.find_peaks(evidence_tensor, thresh)

    genes = library["gene_id"].values
    barcode_locations["target"] = barcode_locations["j"].apply(lambda x: genes[x])
    barcode_locations = barcode_locations.rename(
        columns={"m0": "z", "m1": "y", "m2": "x"}
    )

    return barcode_locations[["x", "y", "z", "target"]]
