from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import nn


class DropPath(nn.Module): # Stochastic Depth（随机丢层） 稳定深网络训练  
# DropPath 只作用在残差连接 x+f(x) 中的f(x)而不是整个x f(x)=0 x不加上任何东西 相当于被跳过了 (整个 block 对这个样本不起作用)
# DropPath 相当于：每次训练用一个“子网络” 模型 ensemble 效果
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob) # 把传进来的 drop_prob 强制转换成 float 类型

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: # not self.training 当前模型处于 推理模式（eval mode）
        # model.train()  → self.training = True  model.eval()   → self.training = False
        # 推理时如果还随机丢弃层会导致输出不稳定
            return x
        keep_prob = 1 - self.drop_prob # 保留层的概率
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # 每个样本（batch）一个随机值，而不是每个元素一个
        # （B,） tuple B is batch size;  (1,) * (x.ndim - 1) Python的tuple乘法  e.g. (1,) * 2  →  (1, 1) 
        #  tuple相加就是直接拼接  (a,) + (b, c) → (a, b, c)    所以 最终结果 (B, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # torch.rand(...) ∈ [0, 1) 均匀分布  randn是正态分布
        random_tensor.floor_() # 向下取整 随机生成 mask（每个样本一个） 0 or 1 决定是否跳过整个路径
        return x / keep_prob * random_tensor # / keep_prob 训练时随机丢，但整体期望和原来一样
        # x.shape =(B,N,D) random_tensor.shape = (B, 1, 1) -> Boardcasting (B, N, D) (从右往左对齐维度)
        


class LayerScale(nn.Module): # x -> gamma * x
    def __init__(self, dim: int, init_values: Optional[float] = 1e-5): # init_values 可以是：一个 float  或者 None
        super().__init__()
        if init_values is None or init_values == 0:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.ones(dim) * init_values) # γ 是一个可学习参数（每个维度一个）
            # x + gamma * f(x) \approx x  模型一开始  几乎是 identity（不改变输入） 防止训练初期不稳定
            # 因为Transformer 很深： 很多层 residual 累加  如果每层一开始都很强： 容易爆炸  attention 不稳定
            # 训练过程中：γ 从 1e-5 → 学习变大  网络逐渐开始利用每一层

    def forward(self, x):
        if self.gamma is None:
            return x
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU, # 这个参数需要是“一个能返回 nn.Module 的函数或类”，默认用 nn.GELU
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features 
        # 因为transformer最终输出一定要和输入维度一样 因为之后有residual connection x+f(x) 后者为transformer的输出 所以要求输入维度=输出维度
        # 所以可以 “or in_features” 代码不说明out_feature也是可以的
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) # 实例化一个类（class） drop是丢弃概率p 是实例化nn.Dropout的时候需要的参数

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x))) #注意MLP只对输入的最后一个维度作修改 (B, *, in_features) → (B, *, hidden_features)  
        x = self.drop(self.fc2(x)) # 在 PyTorch 中，nn.Module (比如Dropout) 都是“可调用对象（callable）” 因为其有__call__函数
        # self.drop(x) 等价于 self.drop.forward(x)
        return x
        # 关于Dropout： 训练时随机把一些元素（GELU 的输出（也就是中间特征））置零 网络的权重和偏置不变 防止神经元过度彼此依赖 强迫模型学习更加鲁棒的表示
        # 同时输入-> 输入/(1-dropout probability)保持期望不变 推理阶段则直接返回x
        # 其数学形式： y=(x*mask)/(1-p) mask in {0,1}  P(mask=1)= 1- p
        # 推理时dropout完全关闭 注意这里是MLP的最后一层 但不是整个模型的最后一层 所以可以有Dropout 模型学的是期望行为


class PatchEmbed(nn.Module): # 不是特征提取 而是把图像变成token（序列）（patch tokenizer）
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 按默认值貌似不是方形的
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, # in channels
            embed_dim, # out channels
            kernel_size=patch_size,
            stride=patch_size,
        ) # 用一个 Conv2d 来实现 patch 切分 +作线性映射 embedding 
        # 假设输入是(B, 3, 224, 224) 根据卷积参数 输出为 (B, embed_dim, 14, 14)  （14 = 224/16） 等于把图像切成一个个patch

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2) 
        # tensor.flatten(start_dim)  从第 start_dim 维开始，把后面的维度全部压平成一个维度 把二维空间变成196个patch
        # 注意 这里的patch 不是原像素的patch 但是还是对应那个空间 patch 的表示（embedding）
        # 经过flatten(2) (B, embed_dim, 14, 14)-> (B, embed_dim, 196)  每个 patch = 一个 token（向量）特征维度为embed_dim
        # 经过transpose --> (B, 196, embed_dim) -> Transformer的token输入序列格式 (B,N,D)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0 # 必须整除 保证 dim = num_heads × head_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # 防止 attention 爆炸 Transformer经典公式里有

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 一次性计算Q K V  输出(B, N, 3*dim)（神经网络作用于最后一维） 再拆成Q, K, V
        # 一次矩阵乘法 更快 GPU更高效 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # C = embedding dim (即dim here)  ViT 可能是(B, 197, 768) 197 = 1 个 cls token + 196 个 patch token
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) # (B, N, C) -> (B, N, 3C)
        # reshape(...)：把那一长条 3C 拆成 “Q/K/V × 多头 × 每头维度”   C=num_heads×head_dim 
        '''
        (B, N, 3, num_heads, head_dim)
        这 5 个维度分别表示：
        B:第几个样本
        N:第几个 token
        3:Q / K / V 三种
        num_heads:第几个注意力头
        head_dim:这个头内部的特征维度
        '''
        qkv = qkv.permute(2, 0, 3, 1, 4) # 把维度顺序改成更适合后面算attention 
        # 因为后面我们想很方便地把 Q、K、V 三者拆开，而且每一个都希望长成：(B, num_heads, N, head_dim) 这样才方便做：q @ k.transpose(-2, -1) 得到 attention score
        q, k, v = qkv.unbind(0) # 沿着第 0 维把 Q、K、V 拆出来 沿着第 0 维拆开，并返回一个 tuple  拆分  
        #Q, K, V: (B, num_heads, N, head_dim) 第一个维度消失
        #! 注意 数据在内存中的排列顺序必须对应 Q/K/V + head 切分 Linear 输出的排列是：[Q1 Q2 ... QD | K1 K2 ... KD | V1 V2 ... VD]
        #! reshape 时默认是 按内存顺序（连续）切分   reshape 默认按原来的线性顺序（row-major）重新分组 #！
        #! 先按前面的维度分割整体 再在得到的小范围内继续分割下一个维度

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        # k.transpose(-2, -1).shape = (B, heads, head_dim, N)
        # attn.shape = (B, heads, N, N)
        # 每个 head 里，每个 token 对所有 token 的注意力分数
        #! 关于多维矩阵乘法： 因为 PyTorch 做的是：batch-wise 矩阵乘法 对每个 (B, head)：单独做一次 (N,d) @ (d,N)
        #! 因此只作用在最后两维，其余维度当作 batch 自动广播
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn) 
        # 把部分attention置零 随机屏蔽部分 token 之间的注意力连接 防止模型过度依赖某些 token 关系（overfitting） 防止 attention 过拟合

        x = attn @ v # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C) # (B, N, heads, head_dim) 把最后两个维度按“线性顺序”拼在一起 先heads循环 每个head内再head_dim维度拼接
        # reshape后每个(B,N)有 head1 || head2 || ... || headH
        x = self.proj_drop(self.proj(x)) # 融合不同 head 的信息 得到更有用的表示 这里输入输出均为(B, N, C) 
        # attention负责信息选择 proj负责信息整合 把这些子空间融合成一个统一表示
        # dropout 作用在 attention 的最终输出特征上 随机把一部分特征维度置 0 这发生在每个 token 的 embedding（C 维）
        # attention 会：动态选择最相关的 token 模型可能学成：只依赖少数几个 token  dropout 强制模型学更鲁棒的表示
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm, # 对每个样本的特征维度做归一化（normalize）
        # x.shape = (B, N, D)   LayerNorm 写成 nn.LayerNorm(D) 表示：对每个 token 的 D 维特征做归一化
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))) # （Pre-Norm）
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
