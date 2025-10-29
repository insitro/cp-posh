import io
import time
from typing import Union

import boto3
import numpy as np
from botocore.config import Config
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from cp_posh.utils import data


class S3Dataset(Dataset):
    """
    Pytorch Dataset for loading images from s3 bucket
    """

    def __init__(self):
        self.s3_client = None

    def get_image(
        self, path, max_attempts=10, wait_sec=2
    ) -> Union[Image.Image, np.ndarray]:
        attempt = 0
        while True:
            try:
                if self.s3_client is None:
                    self.s3_client = boto3.client(
                        "s3",
                        config=Config(retries=dict(max_attempts=max_attempts)),
                    )

                bucket, key = data._get_bucket_key((path))
                s3_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                if key.endswith(".npy"):
                    # read numpy inputs
                    out = np.load(io.BytesIO(s3_obj["Body"].read()), allow_pickle=True)
                elif key.endswith(".png") or key.endswith(".jpg"):
                    # read png images
                    out = Image.open(io.BytesIO(s3_obj["Body"].read())).convert("RGB")
                else:
                    raise ValueError(f"Unsupported file format {key}")

                return out

            except Exception as e:
                time.sleep(wait_sec)
                self.s3_client = None
                if attempt > 1:
                    print(f"Attempt {attempt} ", e)
                attempt += 1

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
