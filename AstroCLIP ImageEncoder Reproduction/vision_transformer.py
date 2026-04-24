from __future__ import annotations

from functools import partial
import math
from typing import Sequence, Tuple, Union
# Sequence 有顺序的一类容器 包括list tuple str  e.g.  Sequence[int] -->  list[int] 或 tuple[int]
# Tuple 固定结构的元组  Union可以是多种类型之一 e.g. Union[int, float] 可以是int或float

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from layers import Mlp, PatchEmbed, Block


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True, #!
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        act_layer=nn.GELU,
        ffn_layer="mlp",
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6) # 预填函数 用于统一之后用到norm_layer的配置

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches # 调用属性

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 表示整张图 是可学习的参数 会学习如何汇聚信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 会自动学习空间结构
        # 加上去的self.num_tokens来自cls_token
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        # 一个 可学习的 token 向量  即一个 token（长度 D）用来“替代被 mask 掉的 patch” “定义一个可学习的缺失信号”
        #! 如果用 0， 模型很容易识别“这是空的” 不会去理解上下文 学不到 meaningful 表示   如果用 random, 噪声导致训练不稳定
        #! 用可学习 token：mask_token 会学成“最佳占位符” (context understanding) 告诉模型：这里缺信息，你要推理”

        # 给每一层 Transformer Block 分配一个 DropPath 概率
        if drop_path_uniform: # 所有层： drop 概率一样
            dpr = [drop_path_rate] * depth
        else: # 各层drop rate线性递增 最后一层值为drop_path_rate 浅层几乎不drop 深层drop多
            # 浅层 学：边缘、局部结构→ 保留（稳定基础特征）  深层 学：语义、组合关系 → 多 drop（正则化，深层更容易过拟合）
            dpr = np.linspace(0, drop_path_rate, depth).tolist()

        if ffn_layer != "mlp":
            raise NotImplementedError("This reference implementation keeps only the MLP FFN path.")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=Mlp,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) 
        # 回忆norm_layer = partial(nn.LayerNorm, eps=1e-6) 相当于 f(x) = nn.LayerNorm(x, eps=1e-6)
        # LayerNorm (x-miu)/sqrt(std^2+epsilon) 相当于是Transformer的数值稳定器
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02) # 初始化不能太小 否则“梯度信号太弱”，模型学不动位置 等于：模型几乎“看不到位置”
        # std=0.02 保持和 embedding 同级别的尺度 让：位置 = 有意义信号
        nn.init.normal_(self.cls_token, std=1e-6) # 初始化要“非常小” 不干扰输入数据分布  初始化不要影响前向传播的分布！
        #否则会主导atention 模型不稳定 之后模型会逐渐学习如何汇聚信息 
        #另外非全零的初始化打破对称性 每个维度 slightly 不一样 训练时可以分化（learn different roles）
        # 打破对称性 = 让不同维度/神经元初始状态不完全一样，从而能够学出不同功能
        # 这里覆盖之前定义的 torch.zeros(...) 原位操作
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6) #同理选取std 和cls_token的共同点是 模型自己引入的“额外 token”
            # “不是来自真实数据的东西，一开始不能干扰模型”  提供微小随机性 
        nn.init.normal_(self.mask_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1 - self.num_register_tokens # 第二维度应该是token数 -1 是cls token 结果是当前输入里真正的 patch token 个数
        N = self.pos_embed.shape[1] - 1 # 剩下patch位置编码的数量 把cls token的位置编码去掉
        if npatch == N and w == h: # 这张图实际得到的 patch 数npatch，是否和模型原本那套 pos_embed 对应的 patch 数N；并且输入图像是否是正方形(默认正方形网格 如果实际不是会影响插值)
        # 一致就直接用，不一致才做位置编码插值。
            pos = self.pos_embed
            if self.num_register_tokens > 0:
                return torch.cat((pos[:, :1], pos[:, 1:]), dim=1)
            return pos
        # npatch != N  后面就要进入真正的插值流程，把旧的 patch 位置编码网格 resize 到新的网格大小。

        # 位置编码插值
        pos_embed = self.pos_embed.float() # 先把位置编码转成 float32 因为后面要做插值，插值通常用 float32 更稳定。
        class_pos_embed = pos_embed[:, 0] # (1, dim)
        patch_pos_embed = pos_embed[:, 1:] # (1, N, dim) N 是原来训练时 patch 的总数
        dim = x.shape[-1] # token embedding 维度

        # 计算当前输入图像对应的 patch 网格 不再是M*M 而是w0 * h0
        w0 = w // self.patch_size
        h0 = h // self.patch_size # 得到当前输入一共会切成多少个 patch
        M = int(math.sqrt(N)) # 假设原始 patch 位置编码是从一个正方形网格来的
        assert N == M * M # 判断假设是否成立
        kwargs = {}
        if self.interpolate_offset: # 指定缩放比例
            sx = float(w0 + self.interpolate_offset) / M # 要放大多少倍 offset 是一个微调项 让插值更“对齐网格中心” → 对齐更平滑 （工程经验）
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy) # 从原始 M×M 网格，按比例缩放到接近 w0×h0
        else:
            kwargs["size"] = (w0, h0) # 没有 offset  就直接指定目标大小 把原始位置编码网格 resize 到 (w0, h0)
        patch_pos_embed = nn.functional.interpolate( # 要求输入是图像格式 (b,c,h,w)
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), # 真正做插值前，先把 一维排开的patch 位置编码从序列还原成二维网格
            mode="bicubic", # (1, dim, M, M)插值成(1, dim, w0, h0) 比较平滑的二维插值方法 双三次插值
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim) # 插值后再变回 token 序列
        # 二维 patch 网格重新展平成一维 token 序列
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        # 把 cls 的位置编码加回去 得到整体的位置编码
        '''
        (1, 196, 768)
        → reshape
        (1, 14, 14, 768)
        → permute
        (1, 768, 14, 14)
        → interpolate
        (1, 768, 16, 16)
        → permute
        (1, 16, 16, 768)
        → reshape
        (1, 256, 768)
        再和 cls 的 (1,1,768) 拼起来：
        (1, 257, 768)

        为什么要这样做?
        pos_embed 学出来时绑定了原始 patch 网格
        新图像大小变了, patch 数也变了
        但位置编码仍然要和 patch 一一对应
        所以必须把 patch 位置编码网格 resize 到新的网格大小
        '''

    def prepare_tokens_with_masks(self, x, masks=None):
        B, _, w, h = x.shape
        x = self.patch_embed(x) # x.shape = (B, N, D) D = embedding dim
        if masks is not None: # masks.shape = (B, N) 每个位置 True → 这个 patch 要被 mask False → 保留原 patch
            x = torch.where(
                masks.unsqueeze(-1), # (B, N, 1) # 实际上每(batch,patch)的mask条件是全部一样的
                self.mask_token.to(x.dtype).unsqueeze(0), # mask_token.shape = (1, D) -> unsqueeze (1, 1, D)-> Boardcast (B, N, D)
                x,
            ) # torch.where(condition, x, y)  按条件逐元素选择：满足条件用 x，否则用 y 
            # 三个输入不用完全同 shape，只要能 broadcast
            # to(x.dtype)把tensor的数据类型转换成和x一样
            '''
            cond.shape = (B, N, 1)
            x.shape    = (1, 1, D)
            y.shape    = (B, N, D)

            自动扩展:
            cond → (B, N, D)
            x    → (B, N, D)

            然后逐元素选择
            '''

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) # x.shape =(B, N, D)
        # expand = 复制（广播）tensor，但不真正拷贝内存(共享内存 快)
        # cls_token.shape (1,1,D)  -> expand -1 表示这一维保持原样 不变 所以得到 （B，1，D） -> cat (B,N+1,D)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(B, -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x) # Final LayerNorm（输出归一化） 注意每一层内部已经做了 LayerNorm（Pre-Norm）
        # 使得CLS token / patch token 都在稳定范围内
        # 也方便下游任务（分类 / 对齐）
        # self.norm = partial(nn.LayerNorm, eps=1e-6)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def get_intermediate_layers(self, x, n: Union[int, Sequence] = 1, reshape=False, return_class_token=False, norm=True):
        x = self.prepare_tokens_with_masks(x)
        outputs = []
        total = len(self.blocks) # 列表（nn.ModuleList）推导式生成的 其中每个元素是一个block 这里得到的是总的block的个数
        blocks_to_take = range(total - n, total) if isinstance(n, int) else n # 选取最后n层 从total-n层到total-1层 否则按传入的n （一个Sequence）来求解
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(self.norm(x) if norm else x) # 输出所选各层的结果
        class_tokens = [out[:, 0] for out in outputs] # out.shape = (B, N+1, D) out[:, 0] 取第0个token -> (B, D) (不必写成out[:,0,:])
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]  # patch 输出 [（B，N, D）...]
        if reshape:
            B, _, w, h = x.shape # 原始输入图像尺寸(B,C,H,W) so w=H h=W
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ] # reshape后 把一维 token 排列恢复成二维 patch 网格 因为 N = w // self.patch_siz * h // self.patch_size 最后一列还是D （B, H', W', D）
            # permute之后 变成 （B, D, H', W'） 这是CNN的标准格式 Channel first 
            # .continuous() 保证内存连续（为了后续计算安全）because permute 会打乱内存布局
            '''
            outputs = [
            (B, D, H', W'),   # selected layer1 图像结构 patch token = 图像网格上的一个点
            (B, D, H', W'),   # selected layer2
            ...
            ]
            每一层变成一个feature map
            '''
        if return_class_token:
            return tuple(zip(outputs, class_tokens)) # zip生成迭代器 每个迭代器是tuple
            '''
            (
            (feature_map_layer1, cls1),
            (feature_map_layer2, cls2),
            ...
            )
            '''
        return tuple(outputs) # list转化成tuple (out1, out2, out3, ...)

    def forward(self, x, masks=None, is_training=False):
        ret = self.forward_features(x, masks=masks) # ret是dict
        return ret if is_training else self.head(ret["x_norm_clstoken"])
        # is_training = True 训练模式--> 返回成为完整特征 
        # ret["x_norm_clstoken"]  shape (B, D) 
        #这里的self.head是Identity（方便之后由此接口修改head 占位符）  推理只需要结果

# 根据需要选择不同scale的模型
def vit_small(**kwargs):
    return DinoVisionTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)


def vit_base(**kwargs):
    return DinoVisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)


def vit_large(**kwargs):
    return DinoVisionTransformer(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)


def build_vit_from_cfg(cfg):
    student_cfg = cfg["student"]
    arch = student_cfg["arch"]
    kwargs = dict(
        img_size=cfg["crops"]["global_crops_size"],
        patch_size=student_cfg["patch_size"],
        in_chans=student_cfg.get("in_chans", 3),
        drop_path_rate=student_cfg["drop_path_rate"],
        drop_path_uniform=student_cfg["drop_path_uniform"],
        init_values=student_cfg["layerscale"],
        qkv_bias=student_cfg["qkv_bias"],
        proj_bias=student_cfg["proj_bias"],
        ffn_bias=student_cfg["ffn_bias"],
        ffn_layer=student_cfg["ffn_layer"],
        num_register_tokens=student_cfg.get("num_register_tokens", 0),
        interpolate_antialias=student_cfg.get("interpolate_antialias", False),
        interpolate_offset=student_cfg.get("interpolate_offset", 0.1),
    )
    if arch == "vit_small":
        student = vit_small(**kwargs)
        teacher = vit_small(**kwargs)
        embed_dim = 384
    elif arch == "vit_base":
        student = vit_base(**kwargs)
        teacher = vit_base(**kwargs)
        embed_dim = 768
    elif arch == "vit_large":
        student = vit_large(**kwargs)
        teacher = vit_large(**kwargs)
        embed_dim = 1024
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return student, teacher, embed_dim


'''
Note:  为什么CLS可以代表一个token的类?
因为在 self-attention 中，它可以和所有 patch 交互，并逐层“汇聚全局信息”
对 CLS token 来说：
Q_cls  * 所有 K_patch
得到CLS 对每个 patch 的 attention 权重
然后 CLS = 加权求和(所有 patch 的 V) -> CLS 已经包含全图信息

之后 多层堆叠：
每一层都会: CLS ← 聚合所有 token 慢慢形成更高语义 最终成为整张图的语义表示

patch 会把信息“给 CLS”
attention 是 双向的 CLS 看 patch  patch 也看 CLS 信息在网络中：反复流动 + 聚合 CLS 就像:一个“信息收集器(collector)”

Note: 既然所有 patch 都在 attention 里互相通信，为什么只有 CLS 能代表全局？ 所有 token 都在交换信息 每个 patch 其实也包含“全局信息”
CLS 并不是“天然更强”，而是因为训练目标只用它(只有 CLS 被监督 有直接loss)，模型被迫把全局信息集中到 CLS 上(有用的信息流“被引导” 让 CLS 包含尽可能完整的信息)


'''