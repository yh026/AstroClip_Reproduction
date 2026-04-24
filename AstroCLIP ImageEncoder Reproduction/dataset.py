from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import datasets

try:
    from datasets import load_from_disk # datasets 这个包来自于Hugging Face
except ImportError:
    load_from_disk = None # 防止该包不存在

#! Use

class HFDatasetWrapper(Dataset):
    """Wraps a HuggingFace dataset saved via `datasets.save_to_disk`."""

    def __init__(
        self,
        path: str,
        split: str = "train",
        image_key: str = "image",
        transform: Optional[Callable] = None,
        select_channels=(0, 1, 2),
    ):
        if load_from_disk is None:
            raise ImportError("datasets is required for HFDatasetWrapper")
        ds = load_from_disk(path) # 从本地目录加载 Hugging Face 数据集
        self.dataset = ds[split] if isinstance(ds, dict) else ds
        self.image_key = image_key
        self.transform = transform
        self.select_channels = select_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx] # 先取出一整条样本，通常是一个字典
        image = item[self.image_key]
        image = np.asarray(image, dtype=np.float32) # float32 is standard in DL
        if image.ndim == 3 and image.shape[-1] >= max(self.select_channels) + 1:
            # 得先是“有通道维”的图像  第 0 维至少要有 3 个通道，才能安全地取第 0、1、2 通道 防止越界
            image = image[...,list(self.select_channels)] # image[...,[0, 1, 2]] Select channels (in last dim)
            image = np.transpose(image,(2,0,1)) # (H,W,C)-> (C, H, W)
            # 把多波段图像裁成模型能接受的通道数
           
        image = torch.from_numpy(image)
        if self.transform is not None:
            return self.transform(image)
        return image


def build_dataset(cfg, transform, split="train"):
    dtype = cfg["train"]["dataset_type"]
    path = cfg["train"]["dataset_path"]
    if dtype == "hf_arrow":
        return HFDatasetWrapper(path=path, split=split, transform=transform)
    raise ValueError(f"Unknown dataset_type: {dtype}")


ds = HFDatasetWrapper(
    path="/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10",
    split="train"
)

x = ds[0]
print(type(x))
print(x.shape)
print(x.dtype)

'''
Result:

<class 'torch.Tensor'>
torch.Size([3, 152, 152])
torch.float32

Data Qualified
'''
