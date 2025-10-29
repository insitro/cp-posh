import math
from typing import List, NamedTuple, Optional, cast
import imreg_dft as ird
import numpy as np
from skimage import transform as tf
from cp_posh.registration import plate_geometry


class TransformedCoordinates(NamedTuple):
    source_field_id: int
    source_x: float
    source_y: float

    target_field_id: Optional[int]
    target_x: Optional[float]
    target_y: Optional[float]


def compute_transform(
    target_well: plate_geometry.Well,
    source_well: plate_geometry.Well,
    registration_resolution: int = 1000,
):
    """
    Computes an affine transformation matrix for registration between target well and source well
    using phase correlation between fiduciary marker images

    Parameters
    ----------
    target_well : plate_geometry.Well
        target (reference) well for registration (to)
    source_well : plate_geometry.Well
        source (moving) well for registration (from)
    registration_resolution : int, optional
        the image resolution at which the registration is performed, by default 1000

    Returns
    -------
    tf.SimilarityTransform
        coordinate transformation function (matrix) from wellB to wellA
    """

    if target_well.well_image is None:
        target_well.construct_well_image()
    if source_well.well_image is None:
        source_well.construct_well_image()

    assert target_well.well_image is not None
    assert source_well.well_image is not None

    target_well_lr_image = tf.resize(
        target_well.well_image,
        (registration_resolution, registration_resolution),
        order=3,
        preserve_range=True,
    )
    source_well_lr_image = tf.resize(
        source_well.well_image,
        (registration_resolution, registration_resolution),
        order=3,
        preserve_range=True,
    )

    result = ird.similarity(
        target_well_lr_image / np.percentile(target_well_lr_image, 99),
        source_well_lr_image / np.percentile(source_well_lr_image, 99),
    )

    # mypy cannot statically infer that these are not None, we know they are not because
    # `construct_marker_image`` was called on both wells above
    target_marker_image = cast(np.ndarray, target_well.well_image)
    source_marker_image = cast(np.ndarray, source_well.well_image)
    target_well_lr_scale = target_marker_image.shape[0] / registration_resolution
    source_well_lr_scale = source_marker_image.shape[0] / registration_resolution

    centerY_target = target_well.well_image_height / 2
    centerX_target = target_well.well_image_width / 2

    centerY_source = source_well.well_image_height / 2
    centerX_source = source_well.well_image_width / 2

    translation = result["tvec"]
    rotation = result["angle"]
    scale = result["scale"]

    shift_to_origin = tf.SimilarityTransform(
        translation=(centerX_source, centerY_source)
    )
    scale_to_reg_res = tf.SimilarityTransform(scale=source_well_lr_scale)
    affine_resolution_transform = tf.SimilarityTransform(
        scale=1.0 / scale, rotation=rotation * math.pi / 180
    )
    affine_shift_transform = tf.SimilarityTransform(
        translation=(-translation[1], -translation[0])
    )
    scale_to_reference_res = tf.SimilarityTransform(scale=1.0 / target_well_lr_scale)
    shift_to_reference_origin = tf.SimilarityTransform(
        translation=(-centerX_target, -centerY_target)
    )

    tform = (
        shift_to_reference_origin
        + scale_to_reference_res
        + affine_shift_transform
        + affine_resolution_transform
        + scale_to_reg_res
        + shift_to_origin
    )

    return tform


def query_source_to_target_per_field(
    target_well: plate_geometry.Well,
    source_well: plate_geometry.Well,
    tform: tf.SimilarityTransform,
    source_field_id: int,
    source_x_coordinates: np.ndarray,
    source_y_coordinates: np.ndarray,
) -> List[TransformedCoordinates]:
    """
    query point from wellB to wellA using the transform computed using
    tform = compute_transform(wellA, wellB)

    Parameters
    ----------
    target_well : plate.Well
        target_well -> reference well
    source_well : plate.Well
        source_well -> moving well
    tform : tf.SimilarityTransform
        transform computed using compute_transform function on wellA and wellB
    source_field_id : int
        field ID to query in wellB
    source_x_coordinates : List[float]
        query X coordinates (pixel columns) in source_field_id
    source_y_coordinates : List[float]
        query Y coordinates (pixel rows) in source_field_id

    Returns
    -------
    List[TransformedCoordinates]
        list of transformed coordinates with source coordinate references
    """

    fieldB = plate_geometry.Field(
        source_well.well_dataframe[
            source_well.well_dataframe.field_id == source_field_id
        ],
    )

    source_stage_coordinates_x, source_stage_coordinates_y = fieldB.image_to_stage(
        source_x_coordinates, source_y_coordinates
    )
    source_well_pixel_coordinates_x, source_well_pixel_coordinates_y = (
        source_well.stage_to_image(
            source_stage_coordinates_x, source_stage_coordinates_y
        )
    )

    pixel_well_targets = tform.inverse(
        np.array([source_well_pixel_coordinates_x, source_well_pixel_coordinates_y]).T
    )
    pixel_well_target_x, pixel_well_target_y = (
        pixel_well_targets[:, 0],
        pixel_well_targets[:, 1],
    )

    target_stage_coordinates_x, target_stage_coordinates_y = target_well.image_to_stage(
        pixel_well_target_x, pixel_well_target_y
    )

    assert isinstance(target_stage_coordinates_x, np.ndarray)
    assert isinstance(target_stage_coordinates_y, np.ndarray)
    assert len(target_stage_coordinates_x) == len(target_stage_coordinates_y)

    target_field_coordinates: List[TransformedCoordinates] = []
    for p in range(len(target_stage_coordinates_x)):
        target_field = target_well.get_field_of_view_at_stage_coordinate(
            target_stage_coordinates_x[p], target_stage_coordinates_y[p]
        )

        if target_field is not None:
            target_field_id = target_field.field_id
            [target_x], [target_y] = target_field.stage_to_image(
                np.array([target_stage_coordinates_x[p]]),
                np.array([target_stage_coordinates_y[p]]),
            )
            if not isinstance(target_x, float) or not isinstance(target_y, float):
                raise TypeError(
                    (
                        "Expected floats to initialize TransformedCoordinates with, got "
                        f"{target_x=} and {target_y=}"
                    )
                )
            target_field_coordinates.append(
                TransformedCoordinates(
                    source_field_id=source_field_id,
                    source_x=source_x_coordinates[p],
                    source_y=source_y_coordinates[p],
                    target_field_id=target_field_id,
                    target_x=target_x,
                    target_y=target_y,
                )
            )
        else:
            target_field_coordinates.append(
                TransformedCoordinates(
                    source_field_id=source_field_id,
                    source_x=source_x_coordinates[p],
                    source_y=source_y_coordinates[p],
                    target_field_id=None,
                    target_x=None,
                    target_y=None,
                )
            )

    return target_field_coordinates
