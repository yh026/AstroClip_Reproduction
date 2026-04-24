from __future__ import annotations # 让所有类型注解（type hints）延迟解析（变成字符串）

from functools import partial

import torch
from torch import nn

from vision_transformer import build_vit_from_cfg
from dino_head import DINOHead
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss

#! 补充 （导入预训练权重）
import re
from collections import OrderedDict

#! Use

class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_backbone, teacher_backbone, embed_dim = build_vit_from_cfg(cfg) # 这里先执行了模型参数初始化 所以之后导入的预训练backbone权重会覆盖掉
        self.embed_dim = embed_dim

        self.student = nn.ModuleDict({"backbone": student_backbone}) 
        # “装模型的字典” PyTorch 会： 自动注册子模块 参数能被 optimizer 管 .to(device) 自动生效 .train() 自动递归
        self.teacher = nn.ModuleDict({"backbone": teacher_backbone}) 
        # 注意这里不能用nn.Sequential 因为不是顺序结构 而是分叉结构 所以要用ModuleDict

        if cfg["student"].get("pretrained_weights"): # 没取到值就得到None #! 小模型配置文件里有没有指定预训练模型权重 大模型迁移学习/续训有 
            chkpt = torch.load(cfg["student"]["pretrained_weights"], map_location="cpu")
            # 把 .pth 文件读进来，得到一个字典：可能包含 “model” "optimizer" 等等
            # map_location ="cpu" 不管你当前有没有 GPU，都先加载到 CPU  避免 GPU 不匹配报错 更安全（服务器环境常用）
            
            _, mapped = self.load_astrodino_teacher_to_student(self.student["backbone"], chkpt)# 用作者预训练好的teacher权重 → 初始化我的 student
            #! 注意实例方法必须用self.调用 因为其是对象的方法

            # Check dtype mismatch
            model_dict = self.student["backbone"].state_dict()
            flag=0
            for k, v in mapped.items():
                if k in model_dict:
                    if v.dtype != model_dict[k].dtype:
                        flag+=1
                        print("dtype mismatch:", k, v.dtype, model_dict[k].dtype)
            if flag ==0:
                print("No dtype mismatch detected!")

            self.teacher["backbone"].load_state_dict(self.student["backbone"].state_dict(), strict=False) 
            # student的权重拷贝给(初始化)teacher 之后训练时再用 EMA 更新 teacher
            # 之后有teacher梯度更新关闭在if条件以外

            ''' 下面的原代码不适用 就替换了 
            self.student["backbone"].load_state_dict(chkpt["model"], strict=False)
            # strict =False “参数不完全匹配也没关系”: 如果strict=True(默认) 要求： key 完全一致 shape 完全一致 否则直接报错
            # strict=False 允许： 少加载一些层（比如 head 不一样） 多余的参数忽略  backbone 可以复用,head 重新训练
            # 之后可以只加载Vit backnbone 特征提取 不加载任务相关的DINO head/ iBOT head 
            # 用已有的 ViT 预训练模型初始化 student backbone,但允许结构不完全一致
            '''

        self.dino_out_dim = cfg["dino"]["head_n_prototypes"]
        self.do_dino = cfg["dino"]["loss_weight"] > 0 # 如果loss weight是0.0 直接赋值False
        self.do_koleo = cfg["dino"]["koleo_loss_weight"] > 0
        self.do_ibot = cfg["ibot"]["loss_weight"] > 0
        self.ibot_separate_head = cfg["ibot"]["separate_head"] # 逻辑值

        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg["dino"]["head_n_prototypes"],
            hidden_dim=cfg["dino"]["head_hidden_dim"],
            bottleneck_dim=cfg["dino"]["head_bottleneck_dim"],
            nlayers=cfg["dino"]["head_nlayers"],
        )
        self.student["dino_head"] = dino_head()
        self.teacher["dino_head"] = dino_head()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()

        if self.do_ibot: # 如果有ibot损失
            self.ibot_out_dim = ( # 判断是否和DINO用分开的head 还是同一个head
                cfg["ibot"]["head_n_prototypes"]
                if self.ibot_separate_head
                else cfg["dino"]["head_n_prototypes"]
            )
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                ibot_head = partial( # DINO 和 IBOT head的模型结构一样
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg["ibot"]["head_n_prototypes"],
                    hidden_dim=cfg["ibot"]["head_hidden_dim"],
                    bottleneck_dim=cfg["ibot"]["head_bottleneck_dim"],
                    nlayers=cfg["ibot"]["head_nlayers"],
                )
                self.student["ibot_head"] = ibot_head()
                self.teacher["ibot_head"] = ibot_head()

        for p in self.teacher.parameters(): # teacher网络参数不参与梯度优化
            p.requires_grad = False 

    @staticmethod  #! 注意在如果你在类里面定义函数：默认它是“实例方法”，必须有 self； 如果你不想要 self，就必须加 @staticmethod  否则会报错（参数数量不对）
    def load_astrodino_teacher_to_student(student_model, ckpt):
        teacher_state = ckpt["teacher"] # 根据作者的预训练文件确定的key teacher 里面有参数字典

        mapped = OrderedDict() # 空的、有顺序的字典 把旧模型的权重 → 转换成新模型能读的格式

        # key mapping
        for k, v in teacher_state.items(): #
            #! 只加载 backbone，因为它学的是“通用表征”(是“可迁移”的知识)  不加载 head，因为它学的是“任务/损失特定的东西”
            if not k.startswith("backbone."):
                continue

            # 去掉作者backbone参数key最前面的 “backbone”字段
            new_k = k[len("backbone."):]

            # 把 blocks.<group>.<idx>.<rest> -> blocks.<idx>.<rest>
            # <group>应该是之前作者分块计算加上的
            m = re.match(r"^blocks\.(\d+)\.(\d+)\.(.*)$", new_k)
            if m:
                _, flat_idx, rest = m.groups()
                new_k = f"blocks.{flat_idx}.{rest}"

            mapped[new_k] = v

        msg = student_model.load_state_dict(mapped, strict=False) # load_state_dict“期望”的是一个字典类对象（通常是 OrderedDict）
        #! load_state_dict 检查 key 名字（不匹配会报错） tensor shape(不匹配会报错) dtype （不会报错 会自动转换）
        # 所有 key 和 shape 都匹配的参数 → 已经加载进 student_model   不匹配的 → 被跳过（不会报错）
        # 返回的 msg 是：_IncompatibleKeys 对象（PyTorch自动生成） 它里面包含：
        # msg.missing_keys      # 模型需要，但你没提供  msg.unexpected_keys   # 你提供了，但模型不需要

        print("=== preload result ===")
        print("missing keys:")
        for x in msg.missing_keys:
            print("  ", x)

        print("unexpected keys:")
        for x in msg.unexpected_keys:
            print("  ", x)

        return msg, mapped

    def train(self, mode: bool = True):
        # SSLMetaArch 继承自 nn.Module 而 nn.Module 本身就有 .train() 方法  
        # 这里用 super().train() 是在调用 父类的 train 方法
        super().train(mode) # 递归让所有nn.Module的子模块进入 train 模式 包括student和teacher network
        self.teacher.eval() # 把 teacher 改回 eval 模式: 不参与梯度更新 不应该有 dropout / BN 随机性 保证输出稳定
        return self # 注意eval()也会递归

    def forward(self, inputs):
        raise NotImplementedError # “这个类不打算直接用 forward，你必须自己实现它” 但PyTorch 规范要求有 forward 所以必须写
        # 但是forward功能不能满足需要 因此不用 实际上用forward_backward函数替代

    def _teacher_targets(self, global_crops, masks, mask_indices_list, n_masked_patches, upperbound, teacher_temp):
        # 用 teacher 网络提取两个 global view 的特征，然后“交换顺序”，作为 student 的监督目标
        teacher_backbone = self.teacher["backbone"](global_crops, is_training=True) # 输出字典
        teacher_cls_tokens = teacher_backbone["x_norm_clstoken"] # (2B, D)  B: batch size
        teacher_cls_tokens = teacher_cls_tokens.chunk(2) # ((B,D),(B,D)) [VIEW1, VIEW2]
        # 把张量 沿第0维（默认）平均切成2块 注意是按顺序切 不是打乱或者随机
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]), dim=0)# 交换顺序 [VIEW2, VIEW1] （2B, D）
        # 保证teacher和student 不同视角之间做一致性学习（cross-view learning）

        teacher_patch_tokens = teacher_backbone["x_norm_patchtokens"] # （2B, N, D）
        # 这 2B 张图 = 同一批 B 张图的两种增强（global crops） 每张图得到N个patch 每个patch是D维
        dim = teacher_patch_tokens.shape[-1] # 特征维度
        n_cls_tokens = teacher_cls_tokens.shape[0] # 2B

        # 把 teacher 提取出来的 cls token 和被 mask 的 patch token，送进 head，得到 teacher 端的监督目标 logits
        # mask_indices_list: 指定“哪些 patch 被 mask、要拿出来做 iBOT”
        # n_masked_patches: 一共挑出了多少个被 mask patch
        if self.do_ibot and not self.ibot_separate_head: # 要做 iBOT，但 iBOT 不单独用 head，而是和 DINO 共用 dino_head
            buffer_teacher = teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, dim)
            # “临时拼接区”  前面一段放 cls token  后面一段放被选中的 masked patch token
            buffer_teacher[:n_cls_tokens] = teacher_cls_tokens # 前2B行是 cls token
            flat = teacher_patch_tokens.flatten(0, 1) # 拉平成一长串 (2B, N, D) -> (2B*N, D) 把所有 crop 的所有 patch token 排成一个长表
            buffer_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches] = flat.index_select(0, mask_indices_list) 
            # 那条长长的 patch token 表里，只取出 mask 对应的位置 现在 buffer_teacher 里面实际装的是  [ 所有 cls token | 所有被 mask 的 patch token ]
            tokens_after_head = self.teacher["dino_head"](buffer_teacher) # 一次性过同一个 dino_head  因为共用 head，所以 cls 和 patch 一起送进去
            # 输出还是一个二维表，只是每个 token 都被映射到了 prototype/logit 空间
            teacher_cls_after_head = tokens_after_head[:n_cls_tokens]  # 再按位置切回来  给 DINO 用
            teacher_patch_after_head = tokens_after_head[n_cls_tokens : n_cls_tokens + n_masked_patches] # 给 iBOT 用

         # 要做 iBOT，而且 iBOT 单独有自己的 ibot_head
        elif self.do_ibot and self.ibot_separate_head:
            buffer_teacher = teacher_patch_tokens.new_zeros(upperbound, dim) 
            flat = teacher_patch_tokens.flatten(0, 1) # 把第0维和第1维合并成一维 图1各个patch 再图2各个patch
            buffer_teacher[:n_masked_patches] = flat.index_select(0, mask_indices_list) # buffer 里只放 patch token
            # 不再把 cls token 塞进去，因为 cls 和 patch 不共用 head 了
            # x.index_select(dim, index) 按 index 指定的位置挑数据
            teacher_cls_after_head = self.teacher["dino_head"](teacher_cls_tokens)  # cls token 单独走 dino_head
            teacher_patch_after_head = self.teacher["ibot_head"](buffer_teacher)[:n_masked_patches] # masked patch 单独走 ibot_head
            # 最后又切了 [:n_masked_patches]，因为真正有效的只有前面那部分，其余位置只是预留的空位
         # 根本不做 iBOT，只做 DINO
        else: # 只有 cls 分支，没有 patch 分支
            teacher_cls_after_head = self.teacher["dino_head"](teacher_cls_tokens)
            teacher_patch_after_head = None
        
        # teacher logits → 温度缩放 + 去偏移（center）+ softmax → 概率分布（target） 动态更新均值
        if self.cfg["train"]["centering"] == "centering":
            teacher_dino_targets = self.dino_loss.softmax_center_teacher(
                teacher_cls_after_head, teacher_temp=teacher_temp
            ).view(2, -1, teacher_cls_after_head.shape[-1])
            # teacher_cls_after_head.shape = (2B, K) 2B：两个 global view  K：prototype 数量（类似“类别数”）
            # 每个图 → 一个 K维向量（logits）
            # softmax((logits - center) / teacher_temp) 去掉“整体偏移”  控制分布“尖锐程度”  
            # .view()  把 shape 从：(2B, K) 变成： (2, B, K) --> view1: (B, K) view2: (B, K)
            self.dino_loss.update_center(teacher_cls_after_head) # 动态更新 teacher 的“均值偏移”
            # center的作用： 强制：不同 prototype 都要被用到 分布更均匀 避免：所有样本 → 同一个输出
            # 再配合 temperature   teacher 输出变成：稳定 不塌缩 有区分度

            teacher_ibot_targets = None
            if self.do_ibot:
                teacher_patch_after_head = teacher_patch_after_head.unsqueeze(0) 
                # 临时加一个 batch 维度，让 tensor 符合 loss 函数的输入格式
                #  (n_masked_patches, K) 一共有 n_masked_patches 个 patch  每个 patch 是 K 维（prototype logits）
                # 变成 (1, n_masked_patches, K) 因为ibot_patch_loss.softmax_center_teacher(...) 这个函数期望输入是三维的 (batch, n_patches, K)
                teacher_ibot_targets = self.ibot_patch_loss.softmax_center_teacher(
                    teacher_patch_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                ).squeeze(0) # 把这个 batch 维去掉
                self.ibot_patch_loss.update_center(teacher_patch_after_head[:, :n_masked_patches]) #防坍缩
                # teacher 同时服务于 DINO 和 iBOT； teacher 不是“有两个任务”，而是“提供两种监督信号”
        elif self.cfg["train"]["centering"] == "sinkhorn_knopp":
            teacher_dino_targets = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_after_head, teacher_temp=teacher_temp
            ).view(2, -1, teacher_cls_after_head.shape[-1])
            teacher_ibot_targets = None
            if self.do_ibot:
                teacher_ibot_targets = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    teacher_patch_after_head,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=torch.tensor(n_masked_patches, device=global_crops.device),
                )
        else:
            raise NotImplementedError

        return teacher_dino_targets, teacher_ibot_targets

    def forward_backward(self, images, teacher_temp):
        device = next(self.parameters()).device
        # self.parameters() 模型中所有参数的“迭代器” weight1, bias1...
        # next(...) get 第一个参数  .device 得到：cuda:0 或 cpu
        # 这种写法的好处, 永远和模型保持一致
        global_crops = images["collated_global_crops"].to(device, non_blocking=True) # 把数据搬到和模型同一个设备上
        local_crops = images["collated_local_crops"].to(device, non_blocking=True)
        masks = images["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = images["mask_indices_list"].to(device, non_blocking=True)
        n_masked_patches = int(images["n_masked_patches"].item())
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].to(device, non_blocking=True)
        # non_blocking=True 尽量“异步地”把数据从 CPU 拷贝到 GPU，不阻塞程序执行 “边搬数据，边干别的事” 允许并行执行
        # Prerequsite: 数据在 pinned memory（锁页内存）  DataLoader(..., pin_memory=True) pinned memory → GPU DMA 直接读 → 可以异步
        # 所以真正生效条件是： pin_memory=True + non_blocking=True

        n_global_crops = 2
        n_local_crops = self.cfg["crops"]["local_crops_number"]
        # 下面统计“有多少个 loss 项（pair）”，用来做平均/归一化  多个 crop 之间，两两配对算 loss
        n_local_terms = max(n_local_crops * n_global_crops, 1)  # 每个 local crop → 和每个 global crop 对齐 max防止n_local_crops = 0
        n_global_terms = (n_global_crops - 1) * n_global_crops # global crop 之间的“交叉配对数量”
        ibot_loss_scale = 1.0 / n_global_crops 
        # 把 iBOT loss 平均到每个 global view 因为iBOT 只用 global crops  而 DINO：用了 global + local 为了让 loss 量级平衡：iBOT 不要太大
        # DINO loss 是： 所有 crop 对之间的平均 总项数： n_global_terms + n_local_terms
        '''
        local crop 只和 global crop 对齐
        不和 local crop 对齐
        global 也不和自己对齐

        永远是 student 在对齐 teacher 所有 student(local + global) 都只对齐 teacher 的 global 输出
        teacher 只负责提供 target(标准答案)
        student 的角色 去拟合 / 对齐 teacher 的 target
        global student: student(global view) 会对齐 teacher(另一个 global view) (cross view)
        local student: student(local view) 会对齐 teacher(所有 global view)

        本质为：用“全局语义”去监督所有视角
        local crop 学到： 局部 → 全局语义  非常强的表征能力
        global crop 学到： 不同增强 → 一致语义
        '''

        with torch.no_grad(): # 这段代码不参与梯度计算（不构建计算图） teacher 的 forward 完全不参与反向传播
        # no_grad 还会：不存中间梯度 → 显存占用更小
        # teacher 只作为“固定目标生成器”，不参与梯度更新  student loss ← teacher (相当于constant) ← backbone
            teacher_dino_targets, teacher_ibot_targets = self._teacher_targets(
                global_crops,
                masks,
                mask_indices_list,
                n_masked_patches,
                upperbound,
                teacher_temp,
            )

        # 这段在做三件事 对齐并求相应损失
        loss_dict = {}
        loss_accumulator = 0.0

        student_global = self.student["backbone"](global_crops, masks=masks, is_training=True)
        student_local = self.student["backbone"](local_crops, masks=None, is_training=True)

        student_local_cls = self.student["dino_head"](student_local["x_norm_clstoken"])
        student_global_cls = self.student["dino_head"](student_global["x_norm_clstoken"])

        if self.do_ibot:
            flat_student_patch = student_global["x_norm_patchtokens"].flatten(0, 1)
            selected_student_patch = flat_student_patch.index_select(0, mask_indices_list)
            if self.ibot_separate_head:
                student_masked_patch = self.student["ibot_head"](selected_student_patch)
            else:
                student_masked_patch = self.student["dino_head"](selected_student_patch)

        # 用 global teacher 的分布监督所有 student（local + global），并通过多视角配对 + 正则避免 collapse
        # DINO local loss
        if self.do_dino and n_local_crops > 0: #! 这个是附加项 检查：有没有 local crop 可以运行的默认前提是已经在使用DINO了 
        #! 但是为了代码好看且对称还是额外加上了self.do_dino and 
            dino_local = self.dino_loss(
                student_output_list=student_local_cls.chunk(n_local_crops), 
                teacher_out_softmaxed_centered_list=teacher_dino_targets,
            ) / (n_global_terms + n_local_terms) 
            # student_local_cls (n_local_crops * B, K) -> (n_local_crops, B, K)
            # teacher_dino_targets (2, B, K)
            loss_dict["dino_local_crops_loss"] = dino_local
            loss_accumulator = loss_accumulator + self.cfg["dino"]["loss_weight"] * dino_local
            
        # DINO global loss
        if self.do_dino: # DINO global 是“主任务” DINO 的核心 = global ↔ global 所以 如果不用 DINO → global loss 必须关掉
            dino_global = (
                self.dino_loss(
                    student_output_list=[student_global_cls],
                    teacher_out_softmaxed_centered_list=[teacher_dino_targets.flatten(0, 1)],
                )
                * 2
                / (n_global_terms + n_local_terms)
            )
            # student_global_cls (2B, K)   teacher flatten (2, B, K)-> (2B, K)
            # Here * 2  两种方向：G1→G2 G2→G1
            loss_dict["dino_global_crops_loss"] = dino_global
            loss_accumulator = loss_accumulator + self.cfg["dino"]["loss_weight"] * dino_global
            # dino_loss(...) 本质是 

            if self.do_koleo: # 对每个 global view (B, D) 做一个正则：让 embedding 分布更均匀（spread out）
            # 防止：所有 embedding → 聚在一起  collapse 的另一种形式
                cls_tokens = student_global["x_norm_clstoken"]
                koleo_loss = self.cfg["dino"]["koleo_loss_weight"] * sum(
                    self.koleo_loss(p) for p in cls_tokens.chunk(2)
                )
                loss_accumulator = loss_accumulator + koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / 2

        # iBOT 用的是 global crops 里的 patch tokens，尤其是 masked patch tokens
        if self.do_ibot:
            ibot_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_masked_patch, # tudent 对所有被 mask patch 的预测 logits
                    teacher_ibot_targets, # teacher 给出的 patch-level target teacher 端“软标签分布”，是 student 要去拟合的目标
                    student_masks_flat=masks, # mask 信息，告诉 loss：哪些 patch 是被 mask 的，哪些不是。 因为 iBOT 只对 masked patch 计算损失，不会对全部 patch 都算
                    n_masked_patches=n_masked_patches, # 避免把后面 buffer/padding 部分算进去
                    masks_weight=masks_weight, 
                )
                * 2 # 避免把后面 buffer/padding 部分算进去
                * ibot_loss_scale # 2 按 global crop 数量做平均/归一化
            )
            loss_dict["ibot_loss"] = ibot_loss / 2 # 让日志里显示的数值和作者定义的“单方向平均 loss”更一致，方便监控训练
            loss_accumulator = loss_accumulator + self.cfg["ibot"]["loss_weight"] * ibot_loss
            '''
            DINO 是：用 cls token 图像级监督 强调全局语义一致性
            iBOT 是 用 patch token  patch 级监督 强调局部结构建模
            Note: local crop 完全是给 DINO 用的  local crop 是“图像级别”的输入，不是 patch
            patch 是从每个 crop(global/local)内部切出来的 token (把一张图(crop)切成小块) 每个 patch 是：一个 token D维向量
            iBOT 不用 local crop生成patch 因为local crop 已经很小 再做 patch → 信息太少  teacher 会不稳定

            global crop → DINO + iBOT
            local crop → 只给 DINO
            patch token → 只给 iBOT
            '''

        loss_accumulator.backward()
        return loss_dict

    @torch.no_grad()
    def update_teacher(self, m): # m 是 momentum，也就是 EMA 的系数
        for student_module_name in self.student.keys(): # 遍历 student 的每个模块名字
            if student_module_name not in self.teacher: # 判断 teacher 里有没有对应模块
                continue
            for ps, pt in zip( # 把同一个模块里的参数一一配对
                self.student[student_module_name].parameters(),
                self.teacher[student_module_name].parameters(),
            ):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m) # 链式调用
                # pt 是一个 nn.Parameter，也就是可训练参数 pt.data 表示这个参数底层真正存的 tensor 数值
                # 用 .data 的目的就是：直接改数值本身，不让 autograd 把这一步当成计算图的一部分
                # .mul_(m) 原地乘法 pt.data.mul_(m) 等价于 pt.data = pt.data * m  相比之下mul：返回新 tensor，不改原来的
                # .add_(ps.data, alpha=1 - m) 原地加法 等价于 等价于 pt. = pt.data + (1 - m) * ps.data
                # 原地操作更省内存 在大模型里，这样更省显存和内存
                # Note: 函数上面已经有 @torch.no_grad() 所以这里本来就不会记录梯度 再用 .data，本质上就是更明确地表示：这里是直接操作参数值，不让 autograd 插手。
                # 把ps.data 直接换成ps也没有问题

                '''
                Note:
                为什么 DINO 不需要 negative samples?
                因为它用“teacher 产生的分布约束 + centering + temperature”来防止 collapse
                传统方法需要 negative loss 是 让正样本相似 让负样本不相似
                如果没有 negative 所有样本 → 同一个向量 loss 仍然可以很小 这就是 collapse(塌缩)
                所以传统方法必须正样本拉近 + 负样本推远
                '''