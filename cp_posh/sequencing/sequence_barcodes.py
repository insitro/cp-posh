import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def sequence_barcodes_per_fov(
    fov_base_calls_all_cycles_df: pd.DataFrame,
    distance_cutoff: int,
) -> pd.DataFrame:
    """
    Stitch base calls across cycles per reference field of view

    Parameters
    ----------
    fov_base_calls_all_cycles_df : pd.DataFrame
        dataframe of base calls per fov across all cycles
    distance_cutoff : int, optional
        disance cutoff for matching base
        locations across cycles (in pixels), by default 2

    Returns
    -------
    pd.DataFrame
        dataframe of stitched barcodes per field of view
    """

    # Get list of unique cycles in dataframe
    cycles = sorted(fov_base_calls_all_cycles_df.cycle.unique())

    # pick first cycle in a sorted list of cycle IDs as reference cycle
    ref_cycle = cycles[0]
    ref_cycle_base_calls = fov_base_calls_all_cycles_df[
        fov_base_calls_all_cycles_df.cycle == ref_cycle
    ]

    field_id = set(ref_cycle_base_calls.field_id.values).pop()

    reference_barcodes_list = []
    for base_call in ref_cycle_base_calls.itertuples():
        reference_barcodes_list.append(
            {
                "barcode": base_call.base,
                "x": base_call.x,
                "y": base_call.y,
                "field_id": int(field_id),
            }
        )
    barcodes_dataframe = pd.DataFrame(reference_barcodes_list)

    for cycle in cycles[1:]:
        cycle_base_calls = fov_base_calls_all_cycles_df[
            fov_base_calls_all_cycles_df.cycle == cycle
        ]

        # Match base locations across cycles using nearest neighbors
        tree = KDTree(
            np.array([cycle_base_calls.x.values, cycle_base_calls.y.values]).transpose(
                1, 0
            ),
            leaf_size=2,
        )
        dist, indices = tree.query(
            np.array(
                [barcodes_dataframe.x.values, barcodes_dataframe.y.values]
            ).transpose(1, 0),
            k=1,
        )

        updated_barcodes = []
        for i in range(len(barcodes_dataframe)):
            # Append base if a base location matches with reference location
            if dist[i] < distance_cutoff:
                cidx = indices[i, 0]
                updated_barcodes.append(
                    {
                        "barcode": barcodes_dataframe.iloc[i].barcode
                        + cycle_base_calls.iloc[cidx].base,
                        "x": barcodes_dataframe.iloc[i].x,
                        "y": barcodes_dataframe.iloc[i].y,
                        "field_id": int(field_id),
                    }
                )
            else:
                # If no base matches with a reference location in a cycle
                # add a generic base `N` for that cycle
                updated_barcodes.append(
                    {
                        "barcode": barcodes_dataframe.iloc[i].barcode + "N",
                        "x": barcodes_dataframe.iloc[i].x,
                        "y": barcodes_dataframe.iloc[i].y,
                        "field_id": int(field_id),
                    }
                )

        # update reference barcodes dataframe to current
        barcodes_dataframe = pd.DataFrame(updated_barcodes)

    return barcodes_dataframe
