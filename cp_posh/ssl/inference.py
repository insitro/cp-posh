import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib, json, os, sys
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file as load_safetensors

from cp_posh.ssl.backbone.vision_transformer import VisionTransformer
from cp_posh.ssl.data.cp_posh import CP_POSH_Dataset
from cp_posh.ssl.meta_arch.dino import DINO

from torch import nn
import argparse

def get_imagenet_pretrained_model(
    model_str: str,
    in_chans: int = 5,
) -> VisionTransformer:
    """
    Load a pretrained DINO model from torch-hub.

    Parameters
    ----------
    model_str : str
        model name to load from torch hub
    in_chans : int, optional
        number of input channels, by default 5

    Returns
    -------
    VisionTransformer
        pretrained DINO-ViT model reparameterized to take `in_chans` number of input channels
    """
    backbone: VisionTransformer = torch.hub.load(
        "facebookresearch/dino:main", model_str
    )
    conv_layer = backbone.patch_embed.proj
    # Ensure that the model accepts the same number of channels as the benchmark dataset contains.
    # If the number of channels is different, average the weights across all channels of the
    # pretrained model and repeat them for all channels of the benchmark dataset.
    # Assume the torch-hub model is trained on ImageNet, so we use ImageNet-specific normalization.
    num_channels_for_model = conv_layer.in_channels
    if num_channels_for_model != in_chans:
        mean_channel = conv_layer.weight.mean(dim=1, keepdims=True)
        new_weight = mean_channel.repeat(1, in_chans, 1, 1)
        new_weight = new_weight * num_channels_for_model / in_chans
        conv_layer.weight = torch.nn.Parameter(new_weight)
        conv_layer.in_channels = in_chans

    return backbone

def _import_entrypoint(entrypoint: str):
    """
    'your_pkg.module.submodule:ClassName' -> Python class
    """
    mod_path, cls_name = entrypoint.split(":")
    module = importlib.import_module(mod_path)
    return getattr(module, cls_name)

def _load_single_safetensors(fp: str) -> Dict[str, torch.Tensor]:
    return load_safetensors(fp)

def _load_cpdino_from_huggingface(
    repo_id: str,
    name: str,                          # key inside manifest["models"]
    revision: Optional[str] = None,
    *,
    map_location: str | torch.device = "cpu",
    token: Optional[str] = None,
    override_kwargs: Optional[dict] = None,
    strict: bool = True,
):
    manifest_path = hf_hub_download(repo_id=repo_id, filename="manifest.json",
                                    revision=revision, token=token)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if name not in manifest["models"]:
        raise KeyError(f"Variant '{name}' not found. Available: {list(manifest['models'].keys())}")

    spec = manifest["models"][name]
    entrypoint = spec["entrypoint"]
    model_dir = spec["path"]
    fmt = spec.get("format", "safetensors")

    kwargs = dict(spec.get("kwargs", {}))
    if override_kwargs:
        kwargs.update(override_kwargs)

    ModelClass = _import_entrypoint(entrypoint)
    model = ModelClass(**kwargs)

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=[f"{model_dir}/*", "model.safetensors.index.json"],  # index might be in subfolder already
    )

    folder = os.path.join(local_dir, model_dir)
    state = _load_single_safetensors(os.path.join(folder, "model.safetensors"))

    # 4) load weights
    model.load_state_dict(state, strict=strict)
    model.to(map_location)
    model.eval()
    return model

def get_model_cp_dino(model_name: str, in_chans: int = 5) -> nn.Module:
    """
    Get the CP-DINO model from a pretrained checkpoint.

    Parameters
    ----------
    model_name : str
        name of the model checkpoint
        one of [cp-dino-300, cp-dino-1640]
    in_chans : int, optional
        number of input channels, by default 5

    Returns
    -------
    nn.Module
        CP-DINO model
    """
    # huggingface checkpoint
    model = _load_cpdino_from_huggingface(
        repo_id="insitro/cp-posh",
        name=model_name,
        override_kwargs={"in_chans": in_chans},
        strict=True,
    )
    return model


def run_inference_on_DINO_model(
    cell_image_dataframe_path: str,
    model_name: str,
    channel_order_in_model_input_order: List[int],
    normalize_mean_in_model_input_order: List[float],
    normalize_std_in_model_input_order: List[float],
) -> pd.DataFrame:
    """
    Run inference on a DINO model and return the embeddings for each cell.

    Parameters
    ----------
    dataframe_path : str
        path to cell image dataframe
    model_name : str
        name of the model in huggingface
    channel_order_in_model_input_order : List[int]
        ordered index of cellpainting channels in the order
        expected by the model
    normalize_mean_in_model_input_order : List[float]
        normalization mean in the order expected by the model
    normalize_std_in_model_input_order : List[float]
        normalization std in the order expected by the model

    Returns
    -------
    pd.DataFrame
        dataframe of embeddings for each cell
    """
    transform_config = DictConfig(
        {
            "name": "CellAugmentationDino",
            "normalization": {
                "mean": normalize_mean_in_model_input_order,
                "std": normalize_std_in_model_input_order,
            },
            "args": {"local_crops_number": 8, "use_coarse_dropout": True},
        }
    )
    dataset = CP_POSH_Dataset(
        ListConfig([cell_image_dataframe_path]),
        channels=channel_order_in_model_input_order,
        is_train=False,
        transform_cfg=transform_config,
    )
    dl = DataLoader(dataset, num_workers=256, batch_size=512, pin_memory=False)

    model = get_model_cp_dino(
        model_name=model_name, in_chans=len(channel_order_in_model_input_order)
    )
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()

    with torch.no_grad():
        embeddings = []
        cell_ids = []
        for _, batch in tqdm(enumerate(dl), total=len(dl)):
            output = model(batch[0].cuda(non_blocking=True)).detach().cpu()
            embeddings.extend(output.numpy())
            cell_ids.extend(batch[1]["ID"])

    embeddings_array = np.array(embeddings)
    cell_ids = np.array(cell_ids)  # type: ignore

    embeddings_df = pd.DataFrame(embeddings_array)
    embeddings_df["ID"] = cell_ids

    dataframe = pd.read_parquet(cell_image_dataframe_path)

    embeddings_df = embeddings_df.merge(
        dataframe[["ID", "barcode", "gene_id"]], on="ID"
    )
    embeddings_df.columns = embeddings_df.columns.astype(str)
    return embeddings_df


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a DINO model and save embeddings as a parquet file."
    )
    parser.add_argument(
        "--cell_image_dataframe_path", type=str, help="Path to cell image dataframe"
    )
    parser.add_argument("--model_name", type=str, help="Name of the model (in huggingface)")
    parser.add_argument(
        "--channel_order_in_model_input_order",
        type=int,
        nargs="+",
        help="Ordered index of cellpainting channels in the order expected by the model",
    )
    parser.add_argument(
        "--normalize_mean_in_model_input_order",
        type=float,
        nargs="+",
        help="Normalization mean in the order expected by the model",
    )
    parser.add_argument(
        "--normalize_std_in_model_input_order",
        type=float,
        nargs="+",
        help="Normalization std in the order expected by the model",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the output parquet file"
    )

    args = parser.parse_args()

    embeddings_df = run_inference_on_DINO_model(
        args.cell_image_dataframe_path,
        args.model_name,
        args.channel_order_in_model_input_order,
        args.normalize_mean_in_model_input_order,
        args.normalize_std_in_model_input_order,
    )

    embeddings_df.to_parquet(args.output_path)


if __name__ == "__main__":
    main()
