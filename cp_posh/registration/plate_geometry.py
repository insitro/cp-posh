from functools import partial
from multiprocessing import Pool
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from skimage import transform
from cp_posh.utils import data


def max_project_images(field_image_list: List[np.ndarray]) -> np.ndarray:
    """
    return max projected image

    Parameters
    ----------
    field_images : List[np.ndarray]
        Max project images images in list

    Returns
    -------
    np.ndarray
        max projection image
    """
    max_projected_image = field_image_list[0]
    for image in field_image_list[1:]:
        max_projected_image = np.maximum(max_projected_image, image)

    return max_projected_image


class AxisAlignedFieldOfViewCropInWellImage(NamedTuple):
    """
    data structure to represent axis aligned field of view
    crop with corresponding position in well image

    Parameters
    ----------
    start_column_index: int
        column start position of FOV crop in well image
    start_row_index: int
        row start position of FOV crop in well image
    end_column_index: int
        column end position of FOV crop in well image
    end_row_index: int
        row end position of FOV crop in well image
    """

    start_column_index: int
    start_row_index: int
    end_column_index: int
    end_row_index: int
    axis_aligned_fov_crop: np.ndarray


class Field:
    """
    data structure for oriented field of view image geometry
    physical dimensions must be isotropic i.e. physical size x == physical size y

    NOTE:
    position_x, position_y are considered as the bottom-left corner coordinates of
    the field in the stage's frame of reference

    Parameters
    ----------
    fov_dataframe: pd.DataFrame
        canonical field of view dataframe
    """

    def __init__(
        self,
        fov_dataframe: pd.DataFrame,
    ):
        self.fov_dataframe = fov_dataframe
        self.field_id = self.fov_dataframe.field_id.unique()[0]
        assert len(self.fov_dataframe.field_id.unique()) == 1

        # select first row to copy fov metadata
        row = self.fov_dataframe.iloc[0]
        self.row = row.row
        self.column = row.column
        self.position_x = row.position_x
        self.position_y = row.position_y
        self.raw_image_width = row.image_width
        self.raw_image_height = row.image_height
        self.raw_physical_pixel_size_x_um = row.physical_size_x
        self.raw_physical_pixel_size_y_um = row.physical_size_y
        self.objective_magnification = row.objective_magnification

        assert (
            self.raw_physical_pixel_size_x_um == self.raw_physical_pixel_size_y_um
        ), "To support oriented images, the pixel dimensions MUST be isotropic."

        self.physical_pixel_size_um = self.raw_physical_pixel_size_x_um

    def get_axis_aligned_image_height(self) -> int:
        """
        get height of stage axis aligned image

        Returns
        -------
        int
            axis aligned image height
        """
        rotation = _get_inverse_camera_rotation_transform(
            image_height=self.raw_image_height,
            image_width=self.raw_image_width,
        ).rotation
        return np.round(
            self.raw_image_width * np.abs(np.sin(rotation))
            + self.raw_image_height * np.abs(np.cos(rotation))
        )

    def get_axis_aligned_image_width(self) -> int:
        """
        get width of axis aligned image

        Returns
        -------
        int
            axis aligned image width
        """
        rotation = _get_inverse_camera_rotation_transform(
            image_height=self.raw_image_height,
            image_width=self.raw_image_width,
        ).rotation
        return np.round(
            self.raw_image_height * np.abs(np.sin(rotation))
            + self.raw_image_width * np.abs(np.cos(rotation))
        )

    def _field_to_stage_transform(self) -> transform.ProjectiveTransform:
        """
        Obtain axis-aligned image (image pixel coordinate space) to
        stage (physical spatial coordinate space) transform

        Returns
        -------
        transform.ProjectiveTransform
            the transform to convert from pixel indices to
            spatial stage coordinates.

        """
        T = _get_yflip_transform(image_height=self.raw_image_height)
        T += _get_inverse_camera_rotation_transform(
            image_height=self.raw_image_height,
            image_width=self.raw_image_width,
        )
        T += _get_image_to_stage_transform(
            physical_pixel_size_um=self.physical_pixel_size_um,
            position_x=self.position_x,
            position_y=self.position_y,
        )

        return T

    def stage_to_image(
        self, stage_x: np.ndarray, stage_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        transform stage coordinates (x, y) -> field of view image coordinates
        (column index, row index)

        Parameters
        ----------
        stage_x : np.ndarray
            stage coordinate x
        stage_y : np.ndarray
            stage coordinate y

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (column index, row index)
        """
        T = self._field_to_stage_transform()
        out = T.inverse(np.array([stage_x, stage_y]).T)
        column_index, row_index = out[:, 0], out[:, 1]

        return column_index, row_index

    def image_to_stage(
        self, column_index: np.ndarray, row_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        transform field of view image (column index, row index) -> stage coordinates (x, y)

        Parameters
        ----------
        column_index : np.ndarray
            field of view image pixel column index
        row_index : np.ndarray
            field of view image pixel row index

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (stage x coordinate, stage y coordinate)
        """
        T = self._field_to_stage_transform()
        out = T(np.array([column_index, row_index]).T)
        stage_x, stage_y = out[:, 0], out[:, 1]
        return stage_x, stage_y

    def get_axis_aligned_fov_image(
        self, channel_index: Optional[int] = 0, image_type: str = "chw"
    ) -> np.ndarray:
        """
        get field of view image with axes aligned to the stage
        coordinate frame of reference

        Parameters
        ----------
        channel_index : Optional[int], optional
            index of channel to load
        image_type: str, optional
            image array order, by default "chw"
        Returns
        -------
        np.ndarray
            axis aligned field of view image
        """

        fov_path: str = self.fov_dataframe["path"].unique()[0]
        img = data.read_image(fov_path, image_type=image_type, loaded_type="chw")
        img = img[[channel_index]]  # type: ignore

        angle_in_degrees = (
            _get_inverse_camera_rotation_transform(
                image_height=self.raw_image_height,
                image_width=self.raw_image_width,
            ).rotation
            * 180
            / np.pi
        )

        img = np.array(
            [
                transform.rotate(ch, angle_in_degrees, resize=True, preserve_range=True)
                for ch in img
            ]
        )
        assert (
            img.shape[1] == self.get_axis_aligned_image_height()
        ), f"image shape ({img.shape[1]})\
            does not match estimated image shape ({self.get_axis_aligned_image_height()})"
        assert (
            img.shape[2] == self.get_axis_aligned_image_width()
        ), f"image shape ({img.shape[2]})\
            does not match estimated image shape ({self.get_axis_aligned_image_width()})"
        return img


def _get_view_image(
    field_id: int,
    well_dataframe: pd.DataFrame,
    view_x_min_stage: float,
    view_y_min_stage: float,
    view_width_pixels: int,
    view_height_pixels: int,
    channel_index: Optional[int] = 0,
    scale_down_factor: float = 1.0,
    image_type: str = "chw",
) -> AxisAlignedFieldOfViewCropInWellImage:
    """
    function to get the field image crop corresponding to the field_id
    that can be pasted in its corresponding location to create a view image.

    This function returns the coordinates and the field crop image that forms a view image

    Parameters
    ----------
    field_id : int
        field of view ID
    well_dataframe : pd.DataFrame
        well image dataframe
    view_x_min_stage : float
        view min x coordinate
    view_y_min_stage : float
        view min y coordinate
    view_width_pixels : int
        view width in pixels
    view_height_pixels : int
        view height in pixels
    channel_index : Optional[int]
        channel index to load
    scale_down_factor: int
        factor to scale images down by, default = 1.0
        the field of view image is scaled down by this factor
    image_type: Optional[str], optional
        image array order, by default "chw"

    Returns
    -------
    ViewImage
        View image object with well x, y coordinates and cropped field image view
    """

    field = Field(
        well_dataframe[well_dataframe.field_id == field_id],
    )
    field_view_bottom_left_x_pixels = (
        (field.position_x - view_x_min_stage)
        / (field.physical_pixel_size_um * scale_down_factor)
    ).astype(int)
    field_view_bottom_left_y_pixels = (
        (field.position_y - view_y_min_stage)
        / (field.physical_pixel_size_um * scale_down_factor)
    ).astype(int)

    if scale_down_factor != 1.0:
        field_image = transform.rescale(
            field.get_axis_aligned_fov_image(
                channel_index=channel_index, image_type=image_type
            ),
            1.0 / scale_down_factor,
            channel_axis=0,
            preserve_range=True,
        )
    else:
        field_image = field.get_axis_aligned_fov_image(
            channel_index=channel_index, image_type=image_type
        )

    field_image = field_image[0]

    # get view coordinates
    field_width_pixels = field_image.shape[-1]
    field_height_pixels = field_image.shape[-2]

    crop_x_left_pixels = max(0, -field_view_bottom_left_x_pixels)
    crop_y_top_pixels = max(
        0, field_view_bottom_left_y_pixels + field_height_pixels - view_height_pixels
    )
    crop_x_right_pixels = field_width_pixels + min(
        0, view_width_pixels - field_view_bottom_left_x_pixels - field_width_pixels
    )
    crop_y_bottom_pixels = field_height_pixels + min(0, field_view_bottom_left_y_pixels)

    field_view_image = field_image[
        ...,
        crop_y_top_pixels:crop_y_bottom_pixels,
        crop_x_left_pixels:crop_x_right_pixels,
    ]

    view_x0 = max(0, field_view_bottom_left_x_pixels)
    view_x1 = min(view_x0 + field_view_image.shape[-1], view_width_pixels)

    view_y0 = max(
        0, view_height_pixels - field_view_bottom_left_y_pixels - field_height_pixels
    )
    view_y1 = min(view_y0 + field_view_image.shape[-2], view_height_pixels)

    return AxisAlignedFieldOfViewCropInWellImage(
        start_column_index=view_x0,
        start_row_index=view_y0,
        end_column_index=view_x1,
        end_row_index=view_y1,
        axis_aligned_fov_crop=field_view_image,
    )


class Well:
    """
    data structure for well image geometry
    physical dimensions must be isotropic i.e. physical size x == physical size y

    Parameters
    ----------
    well_dataframe : DataFrame[CanonicalPlateDataFrameSchema]
        dataframe with rows corresponding to a
        single well in an image acquisition
    source_microscope: microscope.SourceMicroscope  # noqa: E501
        source microscope
    image_storage_format: data.ImageStorageFormat
        image storage format
    field_image_orientation: data.ImageOrientation
        field image orientation
    channel_mapping: Dict[ChannelName, str]
        channel mapping from marker to name of channel in acquisition
    eps_um: int
        epsilon allowance in microns (error between coordinates in same line)
    """

    def __init__(
        self,
        well_dataframe: pd.DataFrame,
        eps_um: int = 150,
    ):
        self.well_dataframe = pd.DataFrame(well_dataframe)

        assert len(self.well_dataframe["well_loc"].unique()) == 1
        assert len(self.well_dataframe["image_width"].unique()) == 1
        assert len(self.well_dataframe["image_height"].unique()) == 1

        self.x_max = self.well_dataframe["position_x"].max()
        self.x_min = self.well_dataframe["position_x"].min()
        self.y_max = self.well_dataframe["position_y"].max()
        self.y_min = self.well_dataframe["position_y"].min()

        self.fov_stage_coordinates = (
            self.well_dataframe[["well_loc", "field_id", "position_x", "position_y"]]
            .drop_duplicates("field_id")
            .sort_values("field_id")
        )

        sample_field_of_view = Field(
            self.well_dataframe[
                self.well_dataframe["field_id"]
                == self.well_dataframe["field_id"].unique()[0]
            ],
        )

        # get axis aligned field of view image parameters
        self.physical_pixel_size_um = sample_field_of_view.physical_pixel_size_um
        self.axis_aligned_fov_image_height = (
            sample_field_of_view.get_axis_aligned_image_height()
        )
        self.axis_aligned_fov_image_width = (
            sample_field_of_view.get_axis_aligned_image_width()
        )
        self.objective_magnification = sample_field_of_view.objective_magnification

        # obtain xy displacement between neighboring fovs and estimate overlap ratio
        dx, dy = self._get_median_xy_displacement_between_neighboring_fovs(
            eps_um=eps_um
        )
        if dx is None or dy is None:
            self.overlap_ratio_x = 0.0
            self.overlap_ratio_y = 0.0
        else:
            self.overlap_ratio_x, self.overlap_ratio_y = self._get_overlap_ratio(dx, dy)

        # compute well image width and height
        self.well_image_width = int(
            (np.round((self.x_max - self.x_min) / self.physical_pixel_size_um))
            + self.axis_aligned_fov_image_width
        )
        self.well_image_height = int(
            (np.round((self.y_max - self.y_min) / self.physical_pixel_size_um))
            + self.axis_aligned_fov_image_height
        )

        self.well_image: Optional[np.ndarray] = None

    def _well_to_stage_transform(self) -> transform.ProjectiveTransform:
        """
        Obtain well image (image pixel coordinate space) to
        stage (physical spatial coordinate space) transform

        Returns
        -------
        transform.ProjectiveTransform
            the transform to convert from pixel indices to
            spatial stage coordinates.

        """
        T = _get_yflip_transform(image_height=self.well_image_height)
        T += _get_image_to_stage_transform(
            physical_pixel_size_um=self.physical_pixel_size_um,
            position_x=self.x_min,
            position_y=self.y_min,
        )

        return T

    def image_to_stage(
        self, column_index: np.ndarray, row_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        well image pixel coordinate (column index, row index) -> stage coordinate (x, y)

        Parameters
        ----------
        column_index : np.ndarray
            well pixel coordinate (column index)
        row_index : np.ndarray
            well pixel coordinate (row index)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (stage coordinate x, stage coordinate y)
        """
        T = self._well_to_stage_transform()
        out = T(np.array([column_index, row_index]).T)
        stage_x, stage_y = out[:, 0], out[:, 1]
        return stage_x, stage_y

    def stage_to_image(
        self, stage_x: np.ndarray, stage_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        stage coordinates (x, y) -> well pixel coordinates (column index, row index)

        Parameters
        ----------
        stage_x : np.ndarray
            stage coordinate x
        stage_y : np.ndarray
            stage coordinate y

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (well pixel coordinate column index, well pixel coordinate row index)
        """
        T = self._well_to_stage_transform()
        out = T.inverse(np.array([stage_x, stage_y]).T)
        column_index, row_index = out[:, 0], out[:, 1]

        return column_index, row_index

    def _get_median_xy_displacement_between_neighboring_fovs(
        self, eps_um: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        obtain median x, y displacements between neighboring fovs
        in stage coordinate space (in um)

        Parameters
        ----------
        eps_um : int
            maximum allowed adjacent field calibration error,

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            median x, y displacement between neighboring field of views (in um)
        """

        neighborhood_graph = self.get_neighborhood_graph(eps_um=eps_um)
        fov_x_coordinate_map = self.fov_stage_coordinates.set_index(
            "field_id"
        ).to_dict()["position_x"]
        fov_y_coordinate_map = self.fov_stage_coordinates.set_index(
            "field_id"
        ).to_dict()["position_y"]

        # collect distance in um between positions of neighboring
        # field of view images
        dxs = []
        dys = []
        for fov_id, neighbors in neighborhood_graph.items():
            if neighbors["right"] is not None:
                dxs.append(
                    np.abs(
                        fov_x_coordinate_map[neighbors["right"]]
                        - fov_x_coordinate_map[fov_id]
                    )
                )
            if neighbors["bottom"] is not None:
                dys.append(
                    np.abs(
                        fov_y_coordinate_map[neighbors["bottom"]]
                        - fov_y_coordinate_map[fov_id]
                    )
                )
        dx = np.median(dxs) if dxs else None
        dy = np.median(dys) if dys else None
        return dx, dy

    def _get_overlap_ratio(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        get overlap ratio between consecutive field of views

        Parameters
        ----------
        dx : float
            x dispacement between horizontal neighbors (left-right) FOVs
        dy : float
            y displacement between vertical neighbors (up-down) FOVs

        Returns
        -------
        Tuple[float, float]
            _description_
        """
        overlap_ratio_x = 1 - dx / (
            self.physical_pixel_size_um * self.axis_aligned_fov_image_width
        )
        overlap_ratio_y = 1 - dy / (
            self.physical_pixel_size_um * self.axis_aligned_fov_image_height
        )

        return overlap_ratio_x, overlap_ratio_y

    def get_neighborhood_graph(self, eps_um) -> Dict[int, Dict[str, Optional[int]]]:
        """
        Get neighborhood graph of field images in a well

        Parameters
        ----------
        eps_um : int, optional
            maximum allowed adjacent field calibration error (in um),

        Returns
        -------
        Dict[int, Dict[str, Optional[int]]]
            neighborhood graph of field of views
            for each field_id, contains `top`, `left`, `bottom`, `right` field of view IDs
        """
        neighborhood_graph: Dict[int, Dict[str, Optional[int]]] = {}
        neighbor_x_offset = (
            self.axis_aligned_fov_image_width * self.physical_pixel_size_um + eps_um
        )
        neighbor_y_offset = (
            self.axis_aligned_fov_image_height * self.physical_pixel_size_um + eps_um
        )

        for field_id in self.well_dataframe.field_id.unique():
            neighborhood_graph[field_id] = {}
            field = self.well_dataframe[self.well_dataframe.field_id == field_id].iloc[
                0
            ]
            top = _find_field_within_range(
                self.well_dataframe,
                field.position_x - eps_um,
                field.position_x + eps_um,
                field.position_y,
                field.position_y + neighbor_y_offset,
                refx=field.position_x,
                refy=field.position_y,
            )
            left = _find_field_within_range(
                self.well_dataframe,
                field.position_x - neighbor_x_offset,
                field.position_x,
                field.position_y - eps_um,
                field.position_y + eps_um,
                refx=field.position_x,
                refy=field.position_y,
            )
            bottom = _find_field_within_range(
                self.well_dataframe,
                field.position_x - eps_um,
                field.position_x + eps_um,
                field.position_y - neighbor_y_offset,
                field.position_y,
                refx=field.position_x,
                refy=field.position_y,
            )
            right = _find_field_within_range(
                self.well_dataframe,
                field.position_x,
                field.position_x + neighbor_x_offset,
                field.position_y - eps_um,
                field.position_y + eps_um,
                refx=field.position_x,
                refy=field.position_y,
            )

            neighborhood_graph[field_id]["top"] = top
            neighborhood_graph[field_id]["left"] = left
            neighborhood_graph[field_id]["bottom"] = bottom
            neighborhood_graph[field_id]["right"] = right

        return neighborhood_graph

    def get_field_of_view_at_stage_coordinate(
        self, stage_x: float, stage_y: float
    ) -> Optional[Field]:
        """
        Get field of view geometry object corresponding to the
        stage coordinate position

        Parameters
        ----------
        stage_x : float
            stage coordinate x
        stage_y : float
            stage coordinate y

        Returns
        -------
        Field | None
            field object corresponding to the world coordinate specified
            returns None if there is no field image cooresponding to the queried
            world coordinate
        """
        offset_x = (
            self.axis_aligned_fov_image_width
            * self.physical_pixel_size_um
            * self.overlap_ratio_x
            / 2
        )
        offset_y = (
            self.axis_aligned_fov_image_height
            * self.physical_pixel_size_um
            * self.overlap_ratio_y
            / 2
        )

        field_df: pd.DataFrame = self.well_dataframe[
            ((stage_x - offset_x) >= self.well_dataframe["position_x"])
            & ((stage_y - offset_y) >= self.well_dataframe["position_y"])
            & (
                (
                    stage_x
                    - self.physical_pixel_size_um * self.axis_aligned_fov_image_width
                    + offset_x
                )
                < self.well_dataframe["position_x"]
            )
            & (
                (
                    stage_y
                    - self.physical_pixel_size_um * self.axis_aligned_fov_image_height
                    + offset_y
                )
                < self.well_dataframe["position_y"]
            )
        ]
        if len(field_df) == 0:
            field_df = self.well_dataframe[
                (stage_x >= self.well_dataframe["position_x"])
                & (stage_y >= self.well_dataframe["position_y"])
                & (
                    (
                        stage_x
                        - self.physical_pixel_size_um
                        * self.axis_aligned_fov_image_width
                    )
                    < self.well_dataframe["position_x"]
                )
                & (
                    (
                        stage_y
                        - self.physical_pixel_size_um
                        * self.axis_aligned_fov_image_height
                    )
                    < self.well_dataframe["position_y"]
                )
            ]

        field = None
        if len(field_df["field_id"].unique()) == 1:
            field = Field(
                field_df,
            )

        return field

    def construct_well_image(
        self,
        well_image_dim: int = 4000,
        scale_down_factor: float = 1.0,
        channel_index: Optional[int] = 0,
        image_type: str = "chw",
    ):
        """
        constructs a cropped well image by selecting the `channels` in field of view image
        Optionally applies `fiduciary_marker_fn` to create fiduciary marker
        image for registration

        Parameters
        ----------
        well_image_dim : int, optional
            well image dimension in pixels, by default 4000
        scale_down_factor : float, optional
            factor to scale images down by, default = 1.0
            the field of view image is scaled down by this factor
        channel_index : Optional[int], optional
            index of channel to load, by default 0
        image_type: Optional[str], optional
            image array order, by default
        """

        center_x = (
            self.x_max
            + self.x_min
            + self.axis_aligned_fov_image_width * self.physical_pixel_size_um
        ) / 2
        center_y = (
            self.y_max
            + self.y_min
            + self.axis_aligned_fov_image_height * self.physical_pixel_size_um
        ) / 2

        physical_field_dim_x = (
            self.axis_aligned_fov_image_width * self.physical_pixel_size_um
        )
        physical_field_dim_y = (
            self.axis_aligned_fov_image_height * self.physical_pixel_size_um
        )

        rescaled_physical_pixel_size_um = (
            self.physical_pixel_size_um * scale_down_factor
        )
        physical_crop_dim = well_image_dim * rescaled_physical_pixel_size_um
        marker_df = self.well_dataframe[
            (
                self.well_dataframe.position_x
                > (center_x - physical_field_dim_x - physical_crop_dim / 2)
            )
            & (self.well_dataframe.position_x < (center_x + physical_crop_dim / 2))
            & (
                self.well_dataframe.position_y
                > (center_y - physical_field_dim_y - physical_crop_dim / 2)
            )
            & (self.well_dataframe.position_y < (center_y + physical_crop_dim / 2))
        ]
        marker_x_min = center_x - (physical_crop_dim / 2)
        marker_y_min = center_y - (physical_crop_dim / 2)

        p = Pool()
        field_view_images = p.map(
            partial(
                _get_view_image,
                well_dataframe=marker_df.copy(),
                view_x_min_stage=marker_x_min,
                view_y_min_stage=marker_y_min,
                view_width_pixels=well_image_dim,
                view_height_pixels=well_image_dim,
                channel_index=channel_index,
                scale_down_factor=scale_down_factor,
                image_type=image_type,
            ),
            marker_df.field_id.unique(),
        )

        for view in field_view_images:
            if self.well_image is None:
                if len(view.axis_aligned_fov_crop.shape) == 3:
                    n_channels = view.axis_aligned_fov_crop.shape[0]
                    self.well_image = np.zeros(
                        (n_channels, well_image_dim, well_image_dim)
                    )
                else:
                    self.well_image = np.zeros((well_image_dim, well_image_dim))

            self.well_image[
                ...,
                view.start_row_index : view.end_row_index,
                view.start_column_index : view.end_column_index,
            ] = np.maximum(
                self.well_image[
                    ...,
                    view.start_row_index : view.end_row_index,
                    view.start_column_index : view.end_column_index,
                ],
                view.axis_aligned_fov_crop,
            )


def _get_yflip_transform(image_height: int) -> transform.ProjectiveTransform:
    """
    obtain y down to y up coordinate transform

                            y
    -------> x           ^
    |                    |
    |          --->      |
    |                    |-------> x
    v
    y

    Parameters
    ----------
    image_height: int
        image height in pixels

    Returns
    -------
    transform.ProjectiveTransform
        y down to y up coordinate space transform
    """
    return transform.AffineTransform(
        matrix=np.array([[1, 0, 0], [0, -1, image_height - 1], [0, 0, 1]])
    )


def _get_inverse_camera_rotation_transform(
    image_height: int, image_width: int
) -> transform.AffineTransform:
    """
    Obtain inverse camera rotation transform from the image orientation.

    Parameters
    ----------
    image_height: int
        image height in pixels
    image_width: int
        image width in pixels
    field_image_orientation: data.ImageOrientation
        the field image orientation within a well

    Returns
    -------
    transform.AffineTransform
        inverse camera rotation transform
    """
    T = transform.AffineTransform(
        translation=(-(image_width - 1) / 2, -(image_height - 1) / 2)
    )
    T += transform.AffineTransform(rotation=-np.pi)
    T += transform.AffineTransform(
        translation=((image_width - 1) / 2, (image_height - 1) / 2)
    )
    return transform.AffineTransform(T)


def _get_image_to_stage_transform(
    physical_pixel_size_um: float, position_x: float, position_y: float
) -> transform.ProjectiveTransform:
    """
    Obtain axis-aligned image (image pixel coordinate space) to
    stage (physical spatial coordinate space) transform

    Parameters
    ----------
    physical_pixel_size_um: float
        the physical size of each pixel in micrometers
    position_x: float
        the physical x stage coordinate of the image
    position_y: float
        the physical y stage coordinate of the image

    Returns
    -------
    tf.ProjectiveTransform
        the axis-aligned image to stage transform
    """
    return transform.AffineTransform(
        scale=physical_pixel_size_um, translation=(position_x, position_y)
    )


def _find_field_within_range(
    well_dataframe: pd.DataFrame,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    refx: float,
    refy: float,
) -> Optional[int]:
    """
    obtain field of view (if available) within a range of
    position_x and position_y values. This method is meant only for unique
    field_id searches in range.

    raises AssertionError if multiple FOVs found in range.

    Parameters
    ----------
    well_dataframe: pd.DataFrame
        well image dataframe
    xmin : float
        min position_x
    xmax : float
        max position_x
    ymin : float
        min position_y
    ymax : float
        max position_y
    refx: float
        reference x position, field of view closest to this point is returned
    refy: float
        reference y position, field of view closest to this point is returned

    Returns
    -------
    Optional[int]
        returns field_id if it exists, else returns None
        raises AssertionError if multiple field_ids are found in range
    """
    field_dfs = well_dataframe.drop_duplicates(["field_id"])
    field_df = field_dfs[
        (field_dfs["position_x"] > xmin)
        & (field_dfs["position_x"] < xmax)
        & (field_dfs["position_y"] > ymin)
        & (field_dfs["position_y"] < ymax)
    ]
    if len(field_df["field_id"].unique()) >= 1:
        # find the point closest to the reference point
        field_df["dist"] = np.linalg.norm(
            [field_df["position_x"] - refx, field_df["position_y"] - refy]
        )
        field_df = field_df.sort_values("dist").head(1)
        return field_df["field_id"].unique()[0]
    else:
        return None
