from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from cp_posh.ssl import transformations
from cp_posh.ssl.data.s3dataset import S3Dataset


class CP_POSH_Dataset(S3Dataset):
    """
    Pytorch Dataset for loading CellPaint-POSH images from s3 bucket

    Parameters
    ----------
    cell_image_dataframes : ListConfig
        list of dataframes containing the tiled single cell image paths
    channels : List[int]
        list of channels to load
    is_train : bool
        whether the dataset is used for training or inference
    transform_cfg : DictConfig
        configuration for the data augmentation transformations
    """

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        cell_image_dataframes: ListConfig,
        channels: List[int],
        is_train: bool,
        transform_cfg: DictConfig,
    ) -> None:
        super().__init__()

        # read all the cell image dataframes
        df = pd.concat(
            [pd.read_parquet(path) for path in cell_image_dataframes], ignore_index=True
        )

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])

        if isinstance(channels[0], str):
            # channel is separated by hyphen
            self.channels: torch.Tensor = torch.tensor(
                [int(c) for c in str(channels[0]).split("-")]
            )
        else:
            self.channels = torch.tensor([c for c in channels])

        # mean and std are used as is (based on training statistics)
        # channel order could be different based on the order of channels in this dataset
        # and input to the model
        normalization_mean = [
            transform_cfg.normalization.mean[int(c)] for c in range(len(channels))
        ]
        normalization_std = [
            transform_cfg.normalization.std[int(c)] for c in range(len(channels))
        ]

        self.is_train = is_train
        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

    def __getitem__(self, index):
        img_hwc = self.get_image(self.data_path[index]).transpose(1,2,0) # HWC
        if img_hwc is None:
            return None

        if self.channels is not None:
            # only load the selected channels
            # because we don't have color jitter in our data augmentation
            # NOTE: converting channels to numpy before slicing
            # as slicing numpy array with 1-length torch tensor squeezes the dimension
            img_hwc = np.array(img_hwc)[:, :, self.channels.numpy()]

        img_chw = self.transform(img_hwc)

        channels = np.array(self.channels)
        if self.is_train:
            metadata = {
                "channels": channels,
            }
        else:
            metadata = {
                "ID": self.data_id[index],
                "channels": channels,
            }

        return (
            img_chw,
            metadata,
        )

    def __len__(self) -> int:
        return len(self.data_path)
