from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
# 用来给“主要用来存数据的类”加一个装饰器，“给数据类自动生成构造函数 + 打印 + 比较，让代码更干净”
'''
@dataclass
class Person:
    name: str
    age: int = 10

dataclass 是一个函数，它接收类 → 修改类 → 返回类 会：
1.读取类里的“字段(field)”（带类型注解的变量）
2.动态生成方法
3.挂到类上

常用于配置类 还可以常量配置(@dataclass(frozen=True)) 防止误改参数
'''
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml # 读写 YAML 文件（配置文件格式）

#! Use

def load_config(path: str | os.PathLike) -> Dict[str, Any]: 
    # path可以是str(字符串路径)或Path对象（比如pathlib.Path）
    # Dict key类型为str value类型为Any(不统一)

    with open(path, "r", encoding="utf-8") as f: 
        # encoding="utf-8"：防止中文乱码
        # with：自动关闭文件（非常重要）
        return yaml.safe_load(f) # YAML文件-> Python字典 （输出字典类型是由safe_load函数本身决定 并不是因为Dict类型说明注解）


def set_seed(seed: int) -> None:
    random.seed(seed) # Python随机性
    np.random.seed(seed) # Numpy随机性
    torch.manual_seed(seed) # CPU上 影响CPU上模型参数初始化 数据增强
    torch.cuda.manual_seed_all(seed) # GPU（CUDA设备）上  manual_seed_all 保证多张GPU都有独立的随机数生成器
    # 影响GPU上的随机数生成 torch.randn(3).cuda() dropout 权重初始化（model.cuda()） 数据增强
    torch.backends.cudnn.deterministic = True # cuDNN也有非确定性算法 CUDA Deep neural Newtork library
    torch.backends.cudnn.benchmark = False #True  # cuDNN是NVDIA给GPU加速深度学习的底层库 禁止动态选择最快算法


class CosineScheduler:
    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
    ) -> None:
        self.schedule = np.empty(total_iters, dtype=np.float64) # 初始化数组
        if warmup_iters > 0:
            self.schedule[:warmup_iters] = np.linspace(
                start_warmup_value, base_value, warmup_iters
            )
        iters = np.arange(total_iters - warmup_iters)
        if len(iters) > 0:
            cosine = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iters / max(len(iters), 1)) # max防止除以0
            ) 
            # (0->1)  -> (base -> final) 线性变换 final + (base - final) * smoothing
            # f(x)= (1+cos(x)）/2  x: 0-> pi  f(x): 1 -> 0   单调下降！ 慢-快-慢 探索-收敛-稳定
            # 对应 t: 0 -> T 所以需要x = pi * t/T
            self.schedule[warmup_iters:] = cosine

    def __getitem__(self, idx: int) -> float: # 可以用该类的实例[idx]调用   实例(x)调用__call__(x)
        return float(self.schedule[idx])


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    # Path 把路径字符串 → 转成 Path 对象
    # .mkdir(...) “创建目录" 函数
    # parents=True 如果上级目录（上面很多层级）不存在 也一起创建
    # exist_ok 如果目录已存在 不会报错 程序继续

def compute_scaled_lr(base_lr: float, batch_size: int, scaling_rule: str) -> float:
    if scaling_rule == "linear_wrt_1024":
        return base_lr * batch_size / 1024.0
    if scaling_rule == "sqrt_wrt_1024":
        return base_lr * math.sqrt(batch_size / 1024.0)
    raise ValueError(f"Unknown scaling rule: {scaling_rule}")
    # batch越大 梯度更接近真实梯度 更可靠 可以放大lr 训练加快
    # scaling 的本质是：保持“每个样本的贡献”一致


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property # 让一个“函数”可以像“属性变量”一样访问 
    # @property = “计算属性（computed attribute）”  它不是存储的值 每次访问时动态计算 (实时统计)
    def avg(self) -> float:
        return self.total / max(self.count, 1)
