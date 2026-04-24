from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

#! Use

def get_vit_layer_id(name: str, n_blocks: int) -> int:
    if name.startswith("patch_embed"): 
        return 0
    if name.startswith("cls_token") or name.startswith("pos_embed") or name.startswith("register_tokens"):
        return 0 
    if name.startswith("blocks"):
        parts = name.split(".") # 把字符串按 "." 分割成一个列表 每个元素也是str
        # 如果 try 里面出错了，就执行 except 里的代码
        try:
            return int(parts[1]) + 1 # parts[1] 为层数
        except Exception: 
        # Exception 捕获所有“正常的错误类型” 各种错误是其子类
            return 0 
    return n_blocks + 1


def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float, layerwise_decay: float, patch_embed_lr_mult: float):
    no_decay = []
    decay = []

    n_blocks = getattr(model, "n_blocks", 12) #! 从 model 里取 n_blocks 属性，如果没有就用 12
    # Python 内置函数： getattr(object, "属性名", 默认值)

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: # 只提取参与优化的参数
            continue
        layer_id = get_vit_layer_id(name, n_blocks)
        lr_scale = layerwise_decay ** (n_blocks + 1 - layer_id) # layerwise_decay<1 
        # 越往后lr越大
        if name.startswith("patch_embed"):
            lr_scale *= patch_embed_lr_mult

        use_decay = param.ndim > 1 and not name.endswith(".bias") and "norm" not in name.lower() and "pos_embed" not in name and "cls_token" not in name and "register_tokens" not in name
        # 只有“真正的权重矩阵”才做 weight decay （让参数变小，防止过拟合）   其他参数（bias / norm / embedding 等）都不做
        # 1. ndim > 1 只选“多维参数”（矩阵 排除参数向量和标量） 2. bias ndim=1  LayerNorm weight  ndim = 1  3. LayerNorm / BatchNorm （norm 层参数控制“尺度”，不是特征权重）加decay会破坏归一化效果
        # 4. 位置编码排除 5. CLS token排除 因为是表示向量 不是权重矩阵 
        key = (layer_id, use_decay, lr_scale)
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": weight_decay if use_decay else 0.0,
                "lr": base_lr * lr_scale,
                "lr_scale": lr_scale,
                "is_last_layer": False,
                "name": [],
            }
        groups[key]["params"].append(param)
        groups[key]["name"].append(name)
    return list(groups.values()) # 取出这个字典里所有的“值” 每个值是一个字典如上
    # dict_values（一个“视图对象”） 不是标准 list 而optimizer需要list或dict 所以要list(...)


def build_head_param_groups(head: nn.Module, base_lr: float, weight_decay: float, is_last_layer=False):
    decay_params, no_decay_params = [], []
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim > 1 and not name.endswith(".bias"):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = []
    if decay_params:
        '''
        在 Python 中：
        空容器 = False
        非空容器 = True
        所以可以用list来作为if语句的判断条件
        '''
        groups.append({
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
            "lr_scale": 1.0,
            "is_last_layer": is_last_layer,
        })
    if no_decay_params:
        groups.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr,
            "lr_scale": 1.0,
            "is_last_layer": is_last_layer,
        })
    return groups
