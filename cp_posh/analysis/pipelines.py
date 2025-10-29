from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm
from skimage import transform as tf
from joblib import Parallel, delayed
from cp_posh.analysis.aggregate import aggregate_embeddings_by_gRNA
from cp_posh.analysis.normalize import (
    normalize_embeddings,
    pca_embeddings,
    prepare_embeddings,
    robust_center_scale,
)
from cp_posh.registration import plate_geometry, transforms
from cp_posh.utils import data, scaling
from cp_posh.sequencing import fcn_base_caller, blob_log_base_caller, sequence_barcodes

base_calling_methods = {"fcn": fcn_base_caller, "blob_log": blob_log_base_caller}


def _call_bases(
    sbs_df: pd.DataFrame, cycle: str, base_calling_method: str
) -> pd.DataFrame:
    """
    Call bases for a given cycle using a base calling method

    Parameters
    ----------
    sbs_df : pd.DataFrame
        sbs image dataframe for a given cycle
    cycle : str
        cycle number
    base_calling_method : str
        base calling method to use

    Returns
    -------
    pd.DataFrame
        base calls for a given cycle
    """
    base_calls_cycle = []
    for field_id in tqdm(sbs_df.field_id.unique(), total=len(sbs_df.field_id.unique())):
        sbs_img = data.read_image(
            sbs_df[sbs_df.field_id == field_id].path.iloc[0], image_type="hwc"
        )
        sbs_img = scaling.rescale_intensity(sbs_img)
        bc = base_calling_methods[base_calling_method].call_bases(
            sbs_img[0:4], base_map={1: "C", 2: "A", 3: "T", 4: "G"}
        )
        bc["field_id"] = field_id
        bc["cycle"] = cycle
        base_calls_cycle.append(bc)
    return pd.concat(base_calls_cycle)


def _get_well_geometry_objs(
    cellpaint_df: pd.DataFrame,
    sbs_dfs: Dict[int, pd.DataFrame],
    base_well_image_dim: int = 5000,
) -> Tuple[plate_geometry.Well, Dict[int, plate_geometry.Well]]:
    """
    get well geometry objects for cellpaint and sbs images

    Parameters
    ----------
    cellpaint_df : pd.DataFrame
        cellpaint image dataframe
    sbs_dfs : Dict[int, pd.DataFrame]
        sbs image dataframes for all cycles
    base_well_image_dim : int, optional
        base well image dimension, by default 5000

    Returns
    -------
    Tuple[plate_geometry.Well, Dict[int, plate_geometry.Well]]
        cellpaint well and sbs wells
    """
    cellpaint_well = plate_geometry.Well(cellpaint_df)
    sbs_wells: Dict[int, plate_geometry.Well] = {}
    for cycle, sbs_df in sbs_dfs.items():
        sbs_wells[cycle] = plate_geometry.Well(sbs_df)

    cellpaint_well.construct_well_image(
        int(
            base_well_image_dim
            * cellpaint_well.objective_magnification
            / sbs_wells[1].objective_magnification
        ),
        channel_index=-1,
    )
    for cycle, _ in sbs_wells.items():
        sbs_wells[cycle].construct_well_image(
            base_well_image_dim, channel_index=-1, image_type="hwc"
        )

    return cellpaint_well, sbs_wells


def _project_to_first_cycle(
    base_calls: pd.DataFrame,
    cycle: int,
    sbs_wells: Dict[int, plate_geometry.Well],
    tforms: Dict[int, tf.SimilarityTransform],
) -> pd.DataFrame:
    """
    project base calls to first SBS cycle

    Parameters
    ----------
    base_calls : pd.DataFrame
        base calls for a given cycle
    cycle : int
        cycle number
    sbs_wells : Dict[int, plate_geometry.Well]
        sbs wells for all cycles
    tforms : Dict[int, tf.SimilarityTransform]
        sbs cycle to first cycle transforms

    Returns
    -------
    pd.DataFrame
        base calls projected to first cycle
    """
    pd.options.mode.chained_assignment = None
    bcxs = []
    for source_field_id in tqdm(
        base_calls.field_id.unique(), total=len(base_calls.field_id.unique())
    ):
        field_bc = base_calls[base_calls.field_id == source_field_id]
        bcx = transforms.query_source_to_target_per_field(
            sbs_wells[1],
            sbs_wells[cycle],
            tforms[cycle],
            source_field_id,
            field_bc.x.values,
            field_bc.y.values,
        )
        bcx_df = pd.DataFrame(bcx)
        field_bc["x"] = bcx_df.loc[:, "target_x"].values
        field_bc["y"] = bcx_df.loc[:, "target_y"].values
        field_bc["field_id"] = bcx_df.loc[:, "target_field_id"].values
        field_bc = field_bc.dropna(axis=0, how="any")
        bcxs.append(field_bc)
    return pd.concat(bcxs)


def _project_barcodes_to_cellpaint(
    barcodes: pd.DataFrame,
    cellpaint_well: plate_geometry.Well,
    sbs_wells: Dict[int, plate_geometry.Well],
    sbs_to_cp_tform: tf.SimilarityTransform,
) -> pd.DataFrame:
    """
    project barcodes to cellpaint space

    Parameters
    ----------
    barcodes : pd.DataFrame
        barcodes dataframe
    cellpaint_well : plate_geometry.Well
        cellpaint well
    sbs_wells : Dict[int, plate_geometry.Well]
        sbs wells for all cycles
    sbs_to_cp_tform : tf.SimilarityTransform
        sbs to cellpaint transform

    Returns
    -------
    pd.DataFrame
        barcodes projected to cellpaint space
    """
    pd.options.mode.chained_assignment = None
    bcxs = []
    for source_field_id in tqdm(
        barcodes.field_id.unique(), total=len(barcodes.field_id.unique())
    ):
        field_barcodes = barcodes[barcodes.field_id == source_field_id]
        bcx = transforms.query_source_to_target_per_field(
            cellpaint_well,
            sbs_wells[1],
            sbs_to_cp_tform,
            source_field_id,
            field_barcodes.x.values,
            field_barcodes.y.values,
        )
        bcx_df = pd.DataFrame(bcx)
        field_barcodes["x"] = bcx_df.loc[:, "target_x"].values
        field_barcodes["y"] = bcx_df.loc[:, "target_y"].values
        field_barcodes["field_id"] = bcx_df.loc[:, "target_field_id"].values

        field_barcodes["source_x"] = bcx_df.loc[:, "source_x"].values
        field_barcodes["source_y"] = bcx_df.loc[:, "source_y"].values
        field_barcodes["source_field_id"] = bcx_df.loc[:, "source_field_id"].values
        bcxs.append(field_barcodes)
    barcodes_in_cellpaint_space = pd.concat(bcxs)
    return barcodes_in_cellpaint_space


def call_bases_per_cycle(
    sbs_dfs: Dict[int, pd.DataFrame], base_calling_method: str
) -> Dict[int, pd.DataFrame]:
    """
    Call bases for all cycles using a base calling method

    Parameters
    ----------
    sbs_dfs : Dict[int, pd.DataFrame]
        sbs image dataframes for all cycles
    base_calling_method : str
        base calling method to use, one of ["fcn", "blob_log"]

    Returns
    -------
    Dict[int, pd.DataFrame]
        base calls for all cycles
    """
    jobs = []
    for cycle, sbs_df in sbs_dfs.items():
        jobs.append(delayed(_call_bases)(sbs_df, cycle, base_calling_method))
    return_values = Parallel(n_jobs=10)(jobs)
    base_calls_per_cycle = {
        c + 1: pd.DataFrame(bc) for c, bc in enumerate(return_values)
    }
    return base_calls_per_cycle


def compute_transforms(
    cellpaint_well: plate_geometry.Well,
    sbs_wells: Dict[int, plate_geometry.Well],
    base_well_image_dim: int = 5000,
) -> Tuple[tf.SimilarityTransform, Dict[int, tf.SimilarityTransform]]:
    """
    compute transforms between cellpaint and first sbs well and between sbs wells to first sbs well

    Parameters
    ----------
    cellpaint_well : plate_geometry.Well
        cellpaint well
    sbs_wells : Dict[int, plate_geometry.Well]
        sbs wells for all cycles
    base_well_image_dim : int, optional
        base well image dimension, by default 5000

    Returns
    -------
    Tuple[tf.SimilarityTransform, Dict[int, tf.SimilarityTransform]]
        sbs to cellpaint transform and sbs cycle to first cycle transforms
    """
    sbs_cycle_to_first_cycle_tforms: Dict[int, tf.SimilarityTransform] = {}
    jobs = []
    for cycle, _ in sbs_wells.items():
        jobs.append(
            delayed(transforms.compute_transform)(
                sbs_wells[1],
                sbs_wells[cycle],
                registration_resolution=base_well_image_dim,
            )
        )
    return_values = Parallel(n_jobs=10)(jobs)
    sbs_cycle_to_first_cycle_tforms = {
        c + 1: tf.SimilarityTransform(tform) for c, tform in enumerate(return_values)
    }
    sbs_to_cellpaint_tform = tf.SimilarityTransform(
        transforms.compute_transform(
            cellpaint_well,
            sbs_wells[1],
            registration_resolution=int(
                base_well_image_dim
                * cellpaint_well.objective_magnification
                / sbs_wells[1].objective_magnification
            ),
        )
    )

    return sbs_to_cellpaint_tform, sbs_cycle_to_first_cycle_tforms


def project_base_calls_to_first_cycle_and_sequence_barcodes(
    base_calls_per_cycle: Dict[int, pd.DataFrame],
    sbs_wells: Dict[int, plate_geometry.Well],
    sbs_cycle_to_first_cycle_tforms: Dict[int, tf.SimilarityTransform],
) -> pd.DataFrame:
    """
    project base calls to first cycle and sequence barcodes

    Parameters
    ----------
    base_calls_per_cycle : Dict[int, pd.DataFrame]
        base calls for all cycles
    sbs_wells : Dict[int, plate_geometry.Well]
        sbs wells for all cycles
    sbs_cycle_to_first_cycle_tforms : Dict[int, tf.SimilarityTransform]
        sbs cycle to first cycle transforms

    Returns
    -------
    pd.DataFrame
        sequenced barcodes
    """
    jobs = []
    for cycle, base_calls in base_calls_per_cycle.items():
        jobs.append(
            delayed(_project_to_first_cycle)(
                base_calls, cycle, sbs_wells, sbs_cycle_to_first_cycle_tforms
            )
        )
    return_values = Parallel(n_jobs=10)(jobs)

    base_calls_all_cycles: pd.DataFrame = pd.concat(return_values)

    barcodes = []
    for field_id in base_calls_all_cycles.field_id.unique():
        barcodes.append(
            delayed(sequence_barcodes.sequence_barcodes_per_fov)(
                base_calls_all_cycles[base_calls_all_cycles.field_id == field_id],
                distance_cutoff=5,
            )
        )
    barcodes = Parallel(n_jobs=16, verbose=True)(barcodes)

    return pd.concat(barcodes)


def sequence_and_map_barcodes_to_cellpaint_image(
    cellpaint_df: pd.DataFrame,
    sbs_dfs: Dict[int, pd.DataFrame],
    sgRNA_library: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    sequence and map barcodes to cellpaint image

    Parameters
    ----------
    cellpaint_df : pd.DataFrame
        cellpaint image dataframe
    sbs_dfs : Dict[int, pd.DataFrame]
        sbs image dataframes for all cycles
    sgRNA_library : pd.DataFrame
        sgRNA library dataframe

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        valid codebook matched barcodes sequenced from SBS images and mapped to Cellpaint image coordinate space,
        all barcodes sequenced from SBS images and mapped to Cellpaint image coordinate space
    """
    # create barcodes which are 10bp long prefix of the sgRNA read across SBS cycles
    sgRNA_library["barcode"] = sgRNA_library["sgRNA"].apply(lambda x: x[0:10])

    # get well geometry objects for cellpaint and sbs images
    print("Creating Well Geometry Objects")
    print("===============================")
    cellpaint_well, sbs_wells = _get_well_geometry_objs(cellpaint_df, sbs_dfs)

    # call bases for all cycles using a base calling method
    print("Calling Bases for all Cycles")
    print("===============================")
    base_calls_per_cycle = call_bases_per_cycle(sbs_dfs, "fcn")

    # compute transforms between cellpaint and first sbs well and between sbs wells to first sbs well
    print("Computing Transforms")
    print("====================")
    sbs_to_cellpaint_tform, sbs_cycle_to_first_cycle_tforms = compute_transforms(
        cellpaint_well, sbs_wells
    )

    # project base calls to first cycle and sequence barcodes
    print("Projecting Base Calls to First Cycle and Sequencing Barcodes")
    print("=============================================================")
    barcodes = project_base_calls_to_first_cycle_and_sequence_barcodes(
        base_calls_per_cycle, sbs_wells, sbs_cycle_to_first_cycle_tforms
    )

    # project barcodes to cellpaint space
    print("Projecting Barcodes to Cellpaint Space")
    print("=======================================")
    barcodes_in_cellpaint_space = _project_barcodes_to_cellpaint(
        barcodes, cellpaint_well, sbs_wells, sbs_to_cellpaint_tform
    )

    # map barcodes to sgRNA library to get valid barcodes and gene identities
    print("Mapping Barcodes to sgRNA Library")
    print("=================================")
    valid_barcodes = barcodes_in_cellpaint_space[
        barcodes_in_cellpaint_space.barcode.apply(lambda x: "N" not in x)
    ].merge(sgRNA_library[["barcode", "gene_id"]], on="barcode")

    print("Done!")
    return valid_barcodes, barcodes_in_cellpaint_space


def normalize_and_aggregate_embeddings(
    raw_embeddings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize and aggregate embeddings

    Parameters
    ----------
    raw_embeddings_df : pd.DataFrame
        raw embeddings dataframe

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        normalized embeddings dataframe and sgRNA aggregated embeddings dataframe
    """

    prep_embeddings_df = prepare_embeddings(raw_embeddings_df)
    norm_embeddings_df = normalize_embeddings(
        prep_embeddings_df, robust_center_scale, unperturbed_control="nontargeting"
    )
    whitened_embeddings_df = pca_embeddings(norm_embeddings_df, whiten=True)
    aggregate_embeddings_df = aggregate_embeddings_by_gRNA(whitened_embeddings_df)
    aggregate_embeddings_df.columns = aggregate_embeddings_df.columns.astype(str)

    return norm_embeddings_df, aggregate_embeddings_df
