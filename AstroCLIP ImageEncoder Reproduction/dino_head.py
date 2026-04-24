from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_ # PyTorch 的“截断正态分布初始化函数”，用于初始化神经网络权重
# 普通正态：可能出现很大或很小的极端值  截断正态：超过一定范围的值会被裁掉重新采样 截断范围默认是 [-2σ, 2σ] 默认均值就是 0 默认std为1
'''
作用： 
防止极端值 普通正态可能出现很大权重 → attention 爆炸
提高训练稳定性 尤其在 self-supervised(DINO)里更敏感
保持 early stage 表达“温和” 初始 embedding 不会太激进

_ 结尾是 in-place 操作（直接修改原 tensor)不会返回新的tensor
'''
from torch.nn.utils import weight_norm

#! Use

class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False, # 不要使用 Batch Normalization（批归一化）
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers, in_dim, bottleneck_dim,
            hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias
        )
        self.apply(self._init_weights) 
        # nn.Module.apply(fn) 的作用是：把函数 fn 递归地应用到当前模块以及所有子模块上
        # 因此这句代码的含义： 遍历整个模型，把每一个 layer 都拿出来，执行一遍 _init_weights(layer)
        # Note: apply 是 结构级操作 不是 tensor 操作，不是 forward：它只在模型初始化阶段用 不参与训练过程
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # weight_norm（Weight Normalization）会把一个 Linear 层的权重拆成两部分 会变成两个参数（作为self.last_layer的属性）：weight_v（方向） weight_g（长度）
        # bias=False 如果有 bias破坏“纯 cosine similarity”结构 所以必须去掉
        self.last_layer.weight_g.data.fill_(1)
        # 把所有输出通道的权重“长度”初始化为 1 也就是g为1
        # 总之 把最后一层 Linear 变成“可学习尺度的 cosine similarity 投影层”，并初始化 scale=1，从而保证 DINO 训练稳定、防止 collapse。

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): # 如果这个module是nn.Linear
            trunc_normal_(m.weight, std=0.02) # 权重 → 用 trunc_normal_ 初始化 # std=0.02 Transformer 系列的“经验最优值”
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # bias（如有） → 设为 0
        # Note：Linear 初始化； Conv（PatchEmbed）PyTorch 默认：Kaiming 初始化（He initialization） 一般 不手动改
        # 关于He Initiaization: 让信号在深层网络中不会爆炸或消失  ViT / Transformer 不用 Kaiming (用于ReLU网络 让方差稳定传播)
        # LayerNorm 默认：weight = 1  bias = 0  Attention 里的 qkv / proj 本质也是：nn.Linear
        # DropPath / GELU  没有参数 → 不需要初始化


    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps) # x-> x/(||x||_2 + epslon) 输入的模为1 p指定二范数  eps防止除以0
        x = self.last_layer(x)
        return x
        # 综上 整个线性层变成：y=Wx=g⋅ v/∥v∥ ​⋅x  x 已经单位化 v/∥v∥ 也是单位向量 所以本质变为 y=g⋅cos(θ)
        # 这个 head 本质是在算 cosine similarity
        '''
        为什么这样设计？ 
        1.防止 collapse: 模型必须学“方向信息”，不能只学大小  
        2. 如果不用 weight_norm: Linear 权重可以无限变大 softmax 会变得 extremely sharp 现在:scale 由 g 控制  而且初始是 1 → 稳定
        3. 和 temperature 配合 输出必须“可控范围”

        '''

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias) # 没有非线性层

    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim)) 
        # 对一维特征做 Batch Normalization（批归一化） 把一批数据的每个特征维度“标准化 + 再缩放”  其中 参数 γ（gamma）和 β（beta）在所有 batch 中是一样的（共享的）
        # 加快收敛  稳定梯度  减少对初始化敏感性
        # 在 Transformer / DINO 中通常关闭（use_bn=False），因为它依赖 batch 统计（均值 方差均来自batch 但batch分布变化大），容易导致不稳定 
        # 而且和Transformer里面的LayerNorm冲突
        # Note: LayerNorm 的 γ（gamma）和 β（beta）也是全局共享参数 维数和特征数相同 每个 feature 一个参数  按维度逐元素作用
    layers.append(nn.GELU()) # 第一层
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias)) # 注意这里是MLP 不是DINO head 所以可以开bias
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU()) # Linear -> BN -> GELU() Gaussian Error Linear Unit  GELU(x)=x⋅Φ(x)  Φ   不是简单“开/关”，而是“概率性通过”  Φ(x)：标准正态分布的 CDF（累积分布函数）
        # GELU() 更平滑 → 梯度更稳定   对小值更友好 GELU：不会直接把小负值砍掉 保留信息   经验效果更好

    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers) # 解包 形成顺序网络模型
