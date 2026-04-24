# # datamodule.py
# from typing import List
# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import CenterCrop
# import datasets
# from torch.utils.data.dataloader import default_collate

# from ..astrodino.data.augmentations import ToRGB


# class AstroClipCollator:
#     def __init__(self, center_crop=144, bands: List[str] = ["g","r","z"], m=0.03, Q=20):
#         self.center_crop = CenterCrop(center_crop)
#         self.to_rgb = ToRGB(bands=bands, m=m, Q=Q)

#     def _process_images(self, images):
#         img_outs = []
#         for img in images:
#             rgb_img = torch.tensor(self.to_rgb(img)[None, :, :, :])
#             img_outs.append(rgb_img)
#         images = torch.cat(img_outs, dim=0)
#         images = self.center_crop(images.permute(0, 3, 2, 1))  # B,C,H,W
#         return images

#     def __call__(self, samples):
#         samples = default_collate(samples)
#         samples["image"] = self._process_images(samples["image"])
#         return samples


# class AstroClipDataloader:
#     def __init__(
#         self,
#         path="/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10",
#         batch_size=256,
#         num_workers=8,
#         center_crop=144,
#         columns=["image", "spectrum"]
#     ):
#         self.path = path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.columns = columns
#         self.collate_fn = AstroClipCollator(center_crop=center_crop)

#         self.dataset = datasets.load_from_disk(self.path)
#         self.dataset.set_format(type="torch", columns=self.columns)

#     def train_dataloader(self):
#         return DataLoader(
#             self.dataset["train"],
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             drop_last=True,
#             collate_fn=self.collate_fn,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.dataset["test"],
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             drop_last=True,
#             collate_fn=self.collate_fn,
#         )





# datamodule.py
from typing import List
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import CenterCrop
import datasets
from torch.utils.data.dataloader import default_collate

from ..astrodino.data.augmentations import ToRGB


class AstroClipCollator:
    def __init__(self, center_crop=144, bands: List[str] = ["g","r","z"], m=0.03, Q=20):
        self.center_crop = CenterCrop(center_crop)
        self.to_rgb = ToRGB(bands=bands, m=m, Q=20)

    def _process_images(self, images):
        img_outs = []
        for img in images:
            rgb_img = torch.tensor(self.to_rgb(img)[None, :, :, :])
            img_outs.append(rgb_img)
        images = torch.cat(img_outs, dim=0)
        images = self.center_crop(images.permute(0, 3, 2, 1))  # B,C,H,W
        return images

    def __call__(self, samples):
        samples = default_collate(samples)
        samples["image"] = self._process_images(samples["image"])
        return samples


class AstroClipDataloader:
    def __init__(
        self,
        path="/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10",
        batch_size=256,
        num_workers=8,
        center_crop=144,
        columns=["image", "spectrum"],
        val_ratio=0.1  # ✅ 验证集比例
    ):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.columns = columns
        self.collate_fn = AstroClipCollator(center_crop=center_crop)

        # 1. 加载原始数据集 (只读操作，不会报错)
        raw_dataset = datasets.load_from_disk(self.path)
        full_train_ds = raw_dataset["train"]
        
        # 2. ✅ 使用 PyTorch/Numpy 手动划分索引 (不涉及文件写入)
        total_size = len(full_train_ds)
        indices = np.arange(total_size)
        
        # 固定随机种子，确保每次运行划分一致
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(total_size * (1 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # 使用 Subset 包装，不复制数据，只引用索引
        self.train_dataset = Subset(full_train_ds, train_indices)
        self.val_dataset = Subset(full_train_ds, val_indices)
        
        # 注意：Subset 返回的是原始 dataset 的 item，所以我们需要在 Collator 或 DataLoader 层面处理
        # 但为了保持 set_format 生效，我们最好直接对原始 dataset 设置格式
        # Subset 会透传 __getitem__，所以只要原始 dataset 格式化过，Subset 取出来的就是 Tensor
        full_train_ds.set_format(type="torch", columns=self.columns)
        
        print(f"✅ Data Loaded: Total={total_size}, Train={len(train_indices)}, Val={len(val_indices)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Subset 本身不打乱，DataLoader 打乱的是索引
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        
    # 可选：如果你想在训练结束后测试真正的 test 集，可以添加这个方法
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.raw_dataset["test"], # 需要在 __init__ 保存 raw_dataset
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         drop_last=False,
    #         collate_fn=self.collate_fn,
    #     )