from typing import Callable
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def prepare_embeddings(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    prepare embeddings for normalization

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input embeddings

    Returns
    -------
    pd.DataFrame
        prepared embeddings
    """
    embeddings_df["plate_well"] = embeddings_df.ID.apply(
        lambda x: x.split("_")[0] + "_" + x.split("_")[1]
    )
    embeddings_df["plate"] = embeddings_df.ID.apply(lambda x: x.split("_")[0])
    embeddings_df = embeddings_df.set_index(
        ["barcode", "gene_id", "plate_well", "plate", "ID"]
    )
    return embeddings_df


def clean_cellstats_features(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    drop irrelevant columns from cellstats features

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input cellstats dataframe

    Returns
    -------
    pd.DataFrame
        clean cellstats dataframe
    """
    drop_columns = [
        "nucleus_mask_centroid_x",
        "nucleus_mask_centroid_y",
        "cell_mask_centroid_x",
        "cell_mask_centroid_y",
    ]
    for c in embeddings_df.columns:
        if "pvalue" in c or "moments" in c:
            drop_columns.append(c)

    drop_columns = list(set(drop_columns).intersection(set(embeddings_df.columns)))
    embeddings_df = embeddings_df.drop(drop_columns, axis=1)
    return embeddings_df


def pca_embeddings(embeddings_df: pd.DataFrame, whiten: bool = True) -> pd.DataFrame:
    """
    return pc (optionally whitened) of embeddings

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input embeddings
    whiten: bool, optional
        whether to whiten embeddings, by default True

    Returns
    -------
    pd.DataFrame
        pc (whitened) embeddings
    """
    embeddings_df = pd.DataFrame(
        PCA(n_components=embeddings_df.shape[1], whiten=whiten).fit_transform(
            embeddings_df.values
        ),
        index=embeddings_df.index,
    )
    return embeddings_df


def robust_center_scale(
    embeddings_df: pd.DataFrame,
    controls: pd.DataFrame,
    batch_column: str = "plate_well",
) -> pd.DataFrame:
    """
    robust center scale embeddings

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input embeddings
    controls : pd.DataFrame
        control embeddings
    batch_column : str, optional
        embeddings are normalized per each unique batch_column,
        by default "plate_well"

    Returns
    -------
    pd.DataFrame
        robust center scaled embeddings
    """
    robust_center_scaled_embeddings = []
    for plate_well in embeddings_df.index.get_level_values(batch_column).unique():  # noqa: E501
        well_df = embeddings_df[
            embeddings_df.index.get_level_values(batch_column) == plate_well
        ]
        well_controls = controls[
            controls.index.get_level_values(batch_column) == plate_well
        ]

        well_control_median = np.median(well_controls, axis=0)
        well_control_median_abs_dev = stats.median_abs_deviation(
            well_controls,
            axis=0,
            scale="normal",
        )

        robust_center_scaled_embeddings.append(
            (well_df - well_control_median) / well_control_median_abs_dev
        )
    robust_center_scaled_embeddings_df = pd.concat(robust_center_scaled_embeddings)

    pre_drop_counts = len(robust_center_scaled_embeddings_df)
    robust_center_scaled_embeddings_df = robust_center_scaled_embeddings_df.dropna(
        axis=1, how="any"
    )
    post_drop_counts = len(robust_center_scaled_embeddings_df)

    assert (
        pre_drop_counts == post_drop_counts
    ), f"dropped columns with NaN values, {pre_drop_counts} \
            -> {post_drop_counts}"
    return robust_center_scaled_embeddings_df


def center_scale(
    embeddings_df: pd.DataFrame,
    controls: pd.DataFrame,
    batch_column: str = "plate_well",
) -> pd.DataFrame:
    """
    center scale embeddings

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input embeddings
    controls : pd.DataFrame
        control embeddings
    batch_column : str, optional
        embeddings are normalized per each unique batch_column,
        by default "plate_well"

    Returns
    -------
    pd.DataFrame
        center scaled embeddings
    """
    center_scaled_embeddings = []
    for plate_well in embeddings_df.index.get_level_values(batch_column).unique():  # noqa: E501
        well_df = embeddings_df[
            embeddings_df.index.get_level_values(batch_column) == plate_well
        ]
        well_controls = controls[
            controls.index.get_level_values(batch_column) == plate_well
        ]

        well_control_median = np.mean(well_controls, axis=0)
        well_control_median_abs_dev = np.std(well_controls, axis=0, ddof=1)

        center_scaled_embeddings.append(
            (well_df - well_control_median) / well_control_median_abs_dev
        )
    center_scaled_embeddings_df = pd.concat(center_scaled_embeddings)

    pre_drop_counts = len(center_scaled_embeddings_df)
    center_scaled_embeddings_df = center_scaled_embeddings_df.dropna(axis=1, how="any")
    post_drop_counts = len(center_scaled_embeddings_df)

    assert (
        pre_drop_counts == post_drop_counts
    ), f"dropped columns with NaN values, {pre_drop_counts}\
          -> {post_drop_counts}"
    return center_scaled_embeddings_df


def normalize_embeddings(
    embeddings_df: pd.DataFrame,
    normalization_method: Callable,
    unperturbed_control: str = "nontargeting",
    batch_column: str = "plate_well",
) -> pd.DataFrame:
    """
    Normalize embeddings using a given normalization method

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        input embeddings
    normalization_method : Callable
        normalization method
    unperturbed_control : str, optional
        control gene_id, by default "nontargeting"
    batch_column : str, optional
        column to use for batch effect correction, by default "plate_well"

    Returns
    -------
    pd.DataFrame
        normalized embeddings
    """

    return normalization_method(
        embeddings_df,
        controls=embeddings_df[
            embeddings_df.index.get_level_values("gene_id") == unperturbed_control  # noqa: E501
        ],
        batch_column=batch_column,
    )
