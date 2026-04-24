from __future__ import annotations

import torch
import torch.nn.functional as F # 函数版的神经网络操作 没参数（纯函数） 不可学习 e.g. F.relu(x)
from torch import nn

#! Use
# 这里所有 loss，本质都是交叉熵 teacher 给一个概率分布 student 输出一个分布 → 做 cross entropy
# 为什么 loss 里都用 F  因为loss 计算通常是： 纯数学操作 不需要参数 注意F 里的函数 不会保存状态

class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim)) # 会广播

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        # softmax_center_teacher = 普通概率分布（带去偏）

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float() # 把 tensor 转成 float32（单精度浮点数） 确保后面计算稳定、不会出数值问题
        # 双精度为.double() 半精度为half()  注意.float()会返回一个新的tensor 不会原地修改
        # 在现代训练里 很可能用的是mixed precision（混合精度） 也就是float16（半精度） 但float16 的问题 范围小 精度低 作exp(x)容易overflow(爆) underflow（变零）
        '''
        平时：用 float16 加速训练
        关键步骤：切回 float32 保证稳定
        teacher output 会被softmax / exp / normalization非常敏感 所以需要数值转化 而且teacher是生成target分布 必须稳定且精确
        '''
        # 生成一个矩阵：Q.shape = (B, K) 每一行 sum = 1（每个样本是分布） 每一列 sum ≈ B/K（每个 prototype 被均匀使用） 这叫balanced assignment（平衡分配）
        Q = torch.exp(teacher_output / teacher_temp).t() # teacher_output.shape = (B, K) B = batch size（样本数） K = prototype 数量 
        #每一行是一个样本的 logits .t() 转置 (B, K)-> (K, B)
        B = Q.shape[1]
        K = Q.shape[0]
        Q /= torch.sum(Q) # 所有列元素（属于一个样本）加起来 = 1 相当于变成一个“概率质量”
        for _ in range(n_iterations): #反复行归一化 列归一化
            Q /= torch.sum(Q, dim=1, keepdim=True)  # 行归一化 行(prototype) → 概率分布 （K, B）/ (K,)
            Q /= K # 强制：每个 prototype 总概率 = 1/K  每个 prototype 被均匀使用
            Q /= torch.sum(Q, dim=0, keepdim=True) # 列归一化  列(样本) → 均匀使用
            Q /= B # 每一列 sum = 1/B  强制：每个样本分布 = 合理概率
            # 为什么要循环？ 行归一化 → 破坏列 列归一化 → 破坏行 所以要反复交替：逐渐收敛 最终得到：行和列都满足约束
        Q *= B # 恢复尺度 之前列被归一成： 1/B 现在恢复成： 每列 sum = 1 回到“正常概率分布”
        return Q.t()  # 转置回来 (K, B) → (B, K) 每个样本一个分布
        # sinkhorn_knopp_teacher = 强制均匀分配的“平衡分布”（更严格防 collapse）反复归一化 整个 batch 一起分配 prototype
        # 行列都被约束的概率矩阵（双随机矩阵)

        '''
        sinkhorn 必须用 float32,因为它做“多次指数 + 归一化循环”，极易数值爆炸/消失
        softmax 不需要强制 float,因为 PyTorch 已经帮你做了数值稳定处理 
        F.softmax(...) 实际上是 softmax(x) = exp(x - max(x)) / sum(...) 内部自动做了 减去最大值(log-sum-exp trick) 避免 exp(很大数) → overflow
        因此softmax即使用float16也比较安全
        但是sinkhorn直接exp(x) 没有减去max  反复归一化导致链式误差累积 误差迅速放大
        '''

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        total_loss = 0.0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1) # student用log_softmax 数值更加稳定
            for t in teacher_out_softmaxed_centered_list:
                total_loss -= torch.sum(t * lsm, dim=-1).mean() # NLL
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True) # teacher_output.shape = (N, K) N = batch size（或2B等）
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        # center 是：teacher_output 的期望值（滑动平均） teacher_output - center 相当于 去掉“均值偏置”
        # 为什么要 EMA（而不是直接用当前 batch）因为单个 batch 不稳定 EMA：历史平均 + 当前信息 更稳定


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        # teacher_patch_tokens.shape = (B, N, D) 
        # B = batch size（global crops 总数，一般是 2B）
        # N = patch 数量（每张图的patch数） 
        # D = embedding维度（= patch_out_dim）
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        teacher_output = teacher_output.float() # (n_masked_patches, K)
        Q = torch.exp(teacher_output / teacher_temp).t() 
        B = Q.shape[1] if n_masked_patches_tensor is None else int(n_masked_patches_tensor.item()) 
        # 注意每个 batch 的 masked patch 数量可能不同# 真实的 n_masked_patches
        # n_masked_patches_tensor 是可选参数 有的调用场景会传，有的不会传，所以可能是 None
        K = Q.shape[0]
        Q /= torch.sum(Q)
        for _ in range(n_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        s = student_patch_tokens_masked # student输出 s.shape = (n_masked_patches, K) 每一行： 一个 patch 的 logits（K个prototype）
        t = teacher_patch_tokens_masked # teacher target t.shape = (n_masked_patches, K) 每一行：一个 patch 的概率分布（已经 softmax 过）
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1) 
        # 经过log_softmax (n_masked_patches, K) 形状不变！！ 每行变成log概率分布 逐元素乘法并不改变形状 .sum之后的shape (n_masked_patches,)

        # 下面给每个 masked patch 分配权重 → 做加权平均 loss（保证每张图贡献均衡）
        # student_masks_flat shape = (B, N)  student_masks_flat 1 表示会被mask（参与loss计算） 0表示不会
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
            # student_masks_flat.sum(-1) 每张图被 mask 的patch数量 (B,) clamp(min=1.0) 防止除以0 每张图的权重取倒数
            # 即使真的mask 0个patch 权重为1 但是因为之后只会选取mask==1的位置 因此也选不到任何patch loss tensor还是空的 所以不影响
            # unsqueeze(-1)  (B,)->(B,1)
            # expand_as -> (B,N) 做 broadcast 扩展（不复制数据 只是view） 每个patch都带上该图的权重  
            # 只选 mask 的位置 [student_masks_flat] [mask]：布尔索引 → 直接变成一维向量(按行展开 布尔索引会包含flatten操作 变成一维)
            # 最终得到shape = (n_masked_patches,)
            # masks_weight 最后是每个 masked patch 的权重 且同一张图的所有 patch 权重相同
            # 原因： 不同图的mask数量不同 如果不加权 多mask的图会影响更大 加权后 每个patch的权重=1/该图mask的数量 保证每张图的总贡献一样
        if n_masked_patches is not None: # 更通用、更保险的 loss API 即使上游多传了一点，我这里也只认前 n_masked_patches 个有效项
            loss = loss[:n_masked_patches] # 截断 因为可能有 padding 或 buffer 只保留 真正有效的 patch
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0] # 每张图的权重和是1 总共B张图

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True) / teacher_patch_tokens.shape[0]
        # teacher_patch_tokens.shape (B, N, K)  B = batch 里有多少个“样本块” 维数和DINO的情况不同 所以不能直接在dim=0上平均
        # .mean() (B,N,K)->(B,K) 每张图的所有patch进行平均
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(nn.Module): # KoLeoLoss = 找最近邻 → 计算距离 → 惩罚距离太小（让点分散）
# student_output.shape = (N, D)  N = token 数（通常是 global cls tokens）  D = embedding 维度
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8) # 用于计算两个向量之间的 L2 距离（欧氏距离 p=2）
        # eps=1e-8 距离为 0 → 数值不稳定（比如 log(0)） 实际计算会变成类似：sqrt(sum(...) + eps)
        # 只计算一一对应的距离 比如x.shape=(N,D) y.shape=(N,D) d=self.pdist(x,y) d.shape = (N,) 因为只计算（x[i], y[i]）之间的成对距离

    def pairwise_NNs_inner(self, x):
        dots = torch.mm(x, x.t()) # 矩阵乘法(N,D)*(D,N)=（N,N） 所有两两点的内积（因为已经 normalize 所以直接就是cos相似度）
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1) # 去掉自己 把对角线（自己和自己）改成-1 否则最近邻永远是自己
        # .view(-1)把矩阵拉平成一维（不复制数据 只是改变形状） （N，N）->(N*N,)    [::(n+1)] Python Slicing [start:end:step] 每隔(n+1)取一个元素
        _, I = torch.max(dots, dim=1) # I.shape = (N,) 每个点的“最相似的另一个点”的索引  torch.max返回 值，指标
        return I

    def forward(self, student_output, eps=1e-8):
        with torch.cuda.amp.autocast(enabled=False): # 关闭 AMP Automatic Mixed Precision（自动混合精度） 强制用float32 因为后面有log(距离) 数值敏感
        # 平时训练： float32 慢但稳定 AMP：float16(快但不稳定)  autocast自动决定哪些算子你用float16 哪些用float32 关闭autocast强制用float32
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1) # 把每个向量变成单位长度向量（都在单位球面上） 之后比较角度 方向即可
            # 做 Lp normalization  eps也是防止除以0用的  把所有点投到单位球面
            I = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[I]) # 计算最近邻距离 (N,D)vs(N,D)=(N,) 第i个元素是x[i] 和 x[I[i]] 的距离 最近邻距离
            return -torch.log(distances + eps).mean() #最近邻距离越大越好 最大化最近邻距离 ≈ 最大化空间覆盖 ≈ 均匀分布
