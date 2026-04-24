from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch

#! Use

class MaskingGenerator:
    def __init__(self, input_size, max_num_patches=None):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.max_num_patches = max_num_patches or self.num_patches
        # “短路逻辑写法” 如果 max_num_patches 有值，就用它；否则(None 0 False "" [])用 self.num_patches

    # 随机选出若干个 patch，把它们标记为 True（mask），其余为 False
    def __call__(self, num_masking_patches):
        num_masking_patches = min(num_masking_patches, self.num_patches) # 防止越界
        mask = torch.zeros(self.num_patches, dtype=torch.bool) # 1D（self.num_patches，）
        idx = torch.randperm(self.num_patches)[:num_masking_patches] 
        # 生成[0,1....,num_patches-1]的随机排列  [:num_masking_patches] 取前num_masking_patches个 [175,3,88...]
        # 相当于随机选num_masking_patches个patch
        mask[idx] = True
        return mask.view(self.height, self.width).flatten()
        # mask对应一个2D patch grid 但模型需要的token是一维序列 所以变成一维
        # 在“空间结构（2D）”和“token序列（1D）”之间做一次显式对齐


def make_mask_for_batch(batch_size, num_patches, mask_ratio_min_max, mask_probability=0.5):
    masks = []
    for _ in range(batch_size):
        if random.random() > mask_probability:
            ratio = 0.0  
        else:
            ratio = random.uniform(*mask_ratio_min_max) 
            # random.uniform(a, b) 在 [a, b] 之间均匀随机采样一个浮点数
            # mask_ratio_min_max 应该是一个两个数值组成的tuple *用于解包得到两个数值
            # 让模型学习在不同强度下都能恢复 使模型更鲁棒
        n = int(num_patches * ratio) # int是向下取整
        mask = torch.zeros(num_patches, dtype=torch.bool)
        if n > 0:
            idx = torch.randperm(num_patches)[:n]
            mask[idx] = True
        masks.append(mask) #[[..],[...],...] 里面每个元素是该batch的mask向量
    return torch.stack(masks, dim=0) # 把多个同形状的 tensor 沿新维度（增加维度）拼成一个 batch （batch_size, num_patches）
    # Note： stack增加一个新维度 cat拼接已有维度

# 把“每个样本里多个crop”整理成模型可用的 batch tensor
def collate_data_and_cast(batch, cfg, num_patches):
    B = len(batch)
    '''
    DataLoader 传进来的是：
    batch = [
    sample1,
    sample2,
    ...,
    sampleB
    ]
    每个 sample 长这样(来自 augmentation—):
    {
    "global_crops": [g1, g2],
    "local_crops": [l1, l2, ..., lN]
    }
    '''
    global_crops = torch.cat(
        [torch.stack([item["global_crops"][0] for item in batch], dim=0), # [g1_sample1, g1_sample2, ..., g1_sampleB] -> stack -> (B, C, H, W)
         torch.stack([item["global_crops"][1] for item in batch], dim=0)], # (B, C, H, W)
        dim=0,
    ) # cat -> (2B, C, H, W) 
    # DINO / iBOT： 每个样本有两个 global view  最终模型输入是：把它们当作 2B 个独立样本
    local_lists = [item["local_crops"] for item in batch]
    '''
    [
    [l1,l2,...,lN],   # sample1
    [l1,l2,...,lN],   # sample2
    ...
    ]
    '''
    local_crops = torch.cat(
        [torch.stack([locals_[i] for locals_ in local_lists], dim=0) for i in range(cfg["crops"]["local_crops_number"])],
        dim=0,
    )
    '''
    1. 固定第 i 个 crop (外层列表推导式固定i值)
    得到 第 i 个 local crop across batch:
    [l_i_sample1, l_i_sample2, ..., l_i_sampleB]

    2. Stack
    (B, C, H, W)

    3. 对所有 i 做
    [local_crops_number个(B, C, H, W)]

    4.cat
    (local_crops_number*B, C, H, W)
    '''

    masks = make_mask_for_batch(
        batch_size=2 * B,
        num_patches=num_patches,
        mask_ratio_min_max=tuple(cfg["ibot"]["mask_ratio_min_max"]), 
        # tuple(list) 直接把list转化为tuple []直接变成()
        mask_probability=cfg["ibot"]["mask_sample_probability"],
    ) # 输出 (2B, num_patches)
    mask_indices_list = masks.flatten().nonzero(as_tuple=False).flatten()
    # masks.flatten()-> (B * num_patches,) 1D
    # x.nonzero() 返回 tensor x中所有“非零（True）元素”的位置索引 
    # as_tuple=False 返回一个二维 tensor，每一行是一个坐标 
    # .nonzero(as_tuple=False)  -> shape = (K, 1)  K = True 的数量 （因为输入是1D tensor 每个index只有一个坐标）
    # flatten -> (K,)
    n_masked_patches = int(mask_indices_list.numel())
    # numel() 返回元素个数-> K

    # 给每一个被 mask 的 patch 分配一个权重，使得每张图的贡献相同（做归一化）
    masks_weight = (
        (1 / masks.sum(-1).clamp(min=1.0)) # (2B,) masks.sum(-1)每张图 mask 了多少个 patch
        .unsqueeze(-1) # (2B, 1)
        .expand_as(masks)[masks] # -> (2B, num_patches) (每一行权重相同) -> (K,)
        # .expand_as(masks)[masks] 是布尔索引  只保留 masks == True 的位置
        .float()
    ) # （K,） 
    # 防止mask越多的图 权重越大 使得每张图总权重为1 贡献一样

    return {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
        "collated_masks": masks,
        "mask_indices_list": mask_indices_list,
        "n_masked_patches": torch.tensor(n_masked_patches, dtype=torch.long), 
        # long（int64） 这个值表示“数量 / 索引相关”，在 PyTorch 里这类数据标准就是 long
        "upperbound": max(n_masked_patches, 1),
        "masks_weight": masks_weight,
    }
