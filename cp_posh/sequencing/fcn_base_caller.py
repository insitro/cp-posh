from typing import Dict
import numpy as np
import torch
import pandas as pd

import abc
from typing import NamedTuple

from skimage import measure

from cp_posh.utils import scaling


class BaseCall(NamedTuple):
    x: int
    y: int
    base: str
    p_C: float
    p_A: float
    p_T: float
    p_G: float
    i_C: float
    i_A: float
    i_T: float
    i_G: float
    ambiguous_base: bool


class SpotDetectionBase(abc.ABC):
    """
    Base class for image spot detection and classification
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def detect_classify_spots(self, x: np.ndarray, **kwargs):
        raise NotImplementedError("method not implemented in sub class")


def _get_max_spot_value(
    image: np.ndarray, x: int, y: int, base_map: Dict[int, str]
) -> Dict[str, float]:
    """
    Get max value at x, y position in a given input image
    """
    values = np.max(
        image[
            :,
            max(0, y - 3) : min(image.shape[1], y + 3),
            max(0, x - 3) : min(image.shape[2], x + 3),
        ],
        axis=(1, 2),
    )
    return {base_map[i]: values[i - 1] for i in base_map.keys()}


def _get_base_call(
    base_index: int,
    x: int,
    y: int,
    pred_proba: np.ndarray,
    sbs_image: np.ndarray,
    base_map: Dict[int, str],
) -> BaseCall:
    """
    Get base call and quality metrics
    """
    spot_base_call = base_map[base_index]

    spot_proba = _get_max_spot_value(pred_proba, x, y, base_map)
    spot_intensity = _get_max_spot_value(sbs_image, x, y, base_map)
    return BaseCall(
        x=x,
        y=y,
        base=spot_base_call,
        p_C=spot_proba["C"],
        p_A=spot_proba["A"],
        p_T=spot_proba["T"],
        p_G=spot_proba["G"],
        i_C=spot_intensity["C"],
        i_A=spot_intensity["A"],
        i_T=spot_intensity["T"],
        i_G=spot_intensity["G"],
        ambiguous_base=False,
    )


def get_base_calls(
    pred_proba: np.ndarray,
    sbs_image: np.ndarray,
    base_map: Dict[int, str],
    threshold: float,
) -> pd.DataFrame:
    """
    Compute base calls and quality from model output probability image

    Parameters
    ----------
    pred_proba : np.ndarray
        predicted spot probability image
    sbs_image : np.ndarray
        input SBS image
    base_map : Dict[int, str]
        index to base mapping
    threshold : float
        spot probability threshold

    Returns
    -------
    pd.DataFrame
        output base calls dataframe
    """

    spots = np.any(pred_proba > threshold, axis=0)
    bases = (np.argmax(pred_proba, axis=0) + 1) * spots
    labeled_spots = measure.label(spots, connectivity=2)
    regions = measure.regionprops(labeled_spots, intensity_image=bases)

    out = []
    for region in regions:
        coords = region.weighted_centroid

        spot_bases = np.unique(
            region.intensity_image[np.nonzero(region.intensity_image)]
        )
        if len(spot_bases) == 1:
            y = int(np.round(coords[0]))
            x = int(np.round(coords[1]))
            spot_base_call = spot_bases[0]
            out.append(
                _get_base_call(spot_base_call, x, y, pred_proba, sbs_image, base_map)
            )
        else:
            for spot_base_call in spot_bases:
                lys, lxs = np.where(region.intensity_image == spot_base_call)
                y = int(np.round(region.bbox[0] + np.mean(lys)))
                x = int(np.round(region.bbox[1] + np.mean(lxs)))
                out.append(
                    _get_base_call(
                        spot_base_call, x, y, pred_proba, sbs_image, base_map
                    )
                )

    return pd.DataFrame(out)


class FCN3SpotDetector(torch.nn.Module):
    """
    3 Layer Fully-Convolutional Network for Spot Detection

    Parameters
    ----------
    n_input_channels: int
        number of input channels (by default 4, for 4 bases)
    n_hidden_channels: int
        hidden channels of network (by default = 64)
    n_out_channels: int
        number of output channels (by default 4, for 4 bases)
    device_selector: str
        device to run model, by default = "cpu"
    """

    def __init__(
        self,
        n_input_channels: int = 4,
        n_hidden_channels: int = 64,
        n_out_channels: int = 4,
        device_selector: str = "cpu",
    ):
        super(FCN3SpotDetector, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, n_hidden_channels, 3, padding=1),
            torch.nn.BatchNorm2d(n_hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, padding=1),
            torch.nn.BatchNorm2d(n_hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_hidden_channels, n_out_channels, 3, padding=1),
        )
        self.device_selector = device_selector
        self.to(self.device_selector)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()  # predict runs in eval mode by default
        pred = torch.nn.Sigmoid()(self.forward(x))
        return pred

    def load_parameters(self, checkpoint_file: str):
        with open(checkpoint_file, "rb") as f:
            checkpoint = torch.load(f, map_location=self.device_selector)
        self.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return x

    def detect_classify_spots(self, x: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        ret = None
        if len(x.shape) == 3:
            ret = self.predict(
                torch.from_numpy(np.array([x])).float().to(self.device_selector),
                **kwargs,
            )[0]
        elif len(x.shape) == 4:
            ret = self.predict(
                torch.from_numpy(x).float().to(self.device_selector), **kwargs
            )
        else:
            raise AssertionError(
                "shape mismatch of input array: expected the input numpy array\
                     of dimension ==  3 or 4"
            )
        return {"bases": ret.detach().cpu().numpy()}


def call_bases(
    image: np.ndarray,
    base_map: Dict[int, str],
    device_selector: str = "cpu",
    threshold: float = 0.5,
    ckpt_path: str = "./checkpoints/fcn_base_caller.pth",
) -> pd.DataFrame:
    """
    call bases using FCN model

    Parameters
    ----------
    image : np.ndarray
        input image (CHW)
    base_map : Dict[str, int]
        base channel mapping dictionary
    device_selector : str, optional
        gpu/cpu device selector, by default "cpu"
    threshold: float, optional
        probability threshold for spot detection, by default 0.5
    ckpt_path: str, optional
        path to checkpoint file, by default "./checkpoints/fcn_base_caller.pth"

    Returns
    -------
    pd.DataFrame
        base call dataframe
    """
    channel_order = ["G", "T", "A", "C"]
    model_base_map = {i + 1: channel_order[i] for i in range(len(channel_order))}
    base_index = {v: k for k, v in base_map.items()}

    sbs_images = []
    for c in channel_order:
        sbs_images.append(image[base_index[c] - 1])

    sbs_image = np.stack(sbs_images)
    sbs_image = scaling.rescale_intensity(sbs_image, 0.1, 99.9)

    spot_detector = FCN3SpotDetector(
        n_input_channels=len(base_map), device_selector=device_selector
    )
    spot_detector.load_parameters(ckpt_path)
    bases = spot_detector.detect_classify_spots(sbs_image)["bases"]
    bases_df = get_base_calls(bases, sbs_image, model_base_map, threshold)

    return bases_df
