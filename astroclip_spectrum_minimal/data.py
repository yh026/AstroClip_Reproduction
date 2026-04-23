from dataclasses import dataclass
from typing import Optional

import datasets
import torch
from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    path: str
    batch_size: int = 64
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "test"
    spectrum_key: str = "spectrum"
    pin_memory: bool = True
    drop_last_train: bool = True


class HFSpectrumDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, spectrum_key: str = "spectrum"):
        self.ds = hf_dataset
        self.spectrum_key = spectrum_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = item[self.spectrum_key]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return {"spectrum": x.float()}


class SpectrumDataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.dataset_dict = None

    def setup(self) -> None:
        self.dataset_dict = datasets.load_from_disk(self.cfg.path)

    def make_loader(self, split: str, shuffle: bool) -> DataLoader:
        ds = HFSpectrumDataset(self.dataset_dict[split], spectrum_key=self.cfg.spectrum_key)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last_train if split == self.cfg.train_split else False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_loader(self.cfg.train_split, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.make_loader(self.cfg.val_split, shuffle=False)
