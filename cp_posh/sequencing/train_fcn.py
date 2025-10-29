import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import retrying

from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from cp_posh.sequencing.fcn_base_caller import FCN3SpotDetector
from cp_posh.utils import data


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation, which is defined as:
    1 - (2 * intersection) / (inputs.sum() + targets.sum())
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        probas = torch.sigmoid(inputs)
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * targets, dims)
        cardinality = torch.sum(probas + targets, dims)
        dice_loss = (2.0 * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss


class SequencingDataset(Dataset):
    def __init__(self, database_df, transform=None, size=256, support=3):
        self.database_df = database_df
        self.transform = transform
        self.size = size
        self.database_df["uid"] = (
            self.database_df.well
            + "-"
            + self.database_df.tile
            + "-"
            + self.database_df.cycle_id.astype(str)
            + "-"
            + self.database_df.tile_id.astype(str)
        )
        self.database_df = self.database_df.set_index("uid")
        self.indices = self.database_df.index.unique()
        self.ksize = support // 2
        self.rsize = support % 2

    def __len__(self):
        return len(self.indices)

    @retrying.retry(stop_max_attempt_number=5)
    def __getitem__(self, idx):
        rows = self.database_df.loc[[self.indices[idx]]]
        assert len(rows.image_path.unique()) == 1

        base_map = {"G": 0, "T": 1, "A": 2, "C": 3}
        bases = np.zeros((4, self.size, self.size))

        img = data.read_image(
            rows.image_path.unique()[0], image_type="hwc", loaded_type="hwc"
        )

        iarr = rows.i.values
        ymin = rows.ymin.values
        xmin = rows.xmin.values
        jarr = rows.j.values
        basearr = rows["base"].values

        for l_ in range(0, len(rows)):
            base = base_map[basearr[l_]]
            y = iarr[l_] - ymin[l_]
            x = jarr[l_] - xmin[l_]
            bases[
                base,
                max(0, y - self.ksize) : min(
                    y + self.ksize + self.rsize, self.size - 1
                ),
                max(0, x - self.ksize) : min(
                    x + self.ksize + self.rsize, self.size - 1
                ),
            ] = 1.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        rv = {"inputs": img, "bases": bases}

        return rv


def collate_for_sequencing(batch):
    batch = [d for d in batch if d["inputs"] is not None]
    if "inputs" in batch[0]:
        return_dict = {}
        for key in batch[0]:
            return_dict[key] = default_collate([d[key] for d in batch])

        return_dict["inputs"] = return_dict["inputs"].float()
    return return_dict


class SequencingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        support_size: int = 3,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.support_size = support_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_transform(self):
        transform = A.Compose(
            [
                # A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.ToFloat(max_value=1),
                ToTensorV2(),
            ]
        )
        return transform

    def get_val_transform(self):
        transform = A.Compose(
            [
                A.ToFloat(max_value=1),
                ToTensorV2(),
            ]
        )
        return transform

    def setup(self, stage=None):
        self.train_dataset = SequencingDataset(
            self.train_df,
            transform=self.get_train_transform(),
            support=self.support_size,
        )
        self.val_dataset = SequencingDataset(
            self.val_df,
            transform=self.get_val_transform(),
            support=self.support_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_for_sequencing,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_for_sequencing,
        )


class POSHSequencingModel(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr=0.01, momentum=0.9, wd=0.0001):
        super().__init__()
        self.model = FCN3SpotDetector(n_channels, 64, n_classes)
        self.loss_fn = DiceLoss()
        self.lr = lr
        self.momentum = momentum
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["inputs"], batch["bases"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["inputs"], batch["bases"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.wd,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2, 4, 6, 8, 10], gamma=0.5
        )
        return [optimizer], [scheduler]


def main(args: argparse.Namespace):
    train_dataframe = data.load_data(args.train_dataframe_path)
    val_dataframe = data.load_data(args.val_dataframe_path)

    data_module = SequencingDataModule(
        train_dataframe,
        val_dataframe,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    n_classes = len(train_dataframe["base"].unique())
    n_channels = n_classes
    model = POSHSequencingModel(n_channels, n_classes, args.lr, args.momentum, args.wd)

    wandb_logger = WandbLogger(project=args.project_name, name=args.experiment_id)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.model_checkpoint, args.experiment_id),
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=args.num_epochs,
        strategy="ddp",
    )

    trainer.fit(model, datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI tool to run a model with a specified dataset"
    )
    parser.add_argument(
        "--train_dataframe_path",
        type=str,
        help="Path to the train dataframe",
    )
    parser.add_argument(
        "--val_dataframe_path",
        type=str,
        help="Path to the validation dataframe",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the model",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for the model",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        help="Weight decay for the model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of the project",
        default="cp_posh_base_calling",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Name of the experiment",
        default="cp_posh_fcn",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Path to save model checkpoints",
        default="./model_checkpoints",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
