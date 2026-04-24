from __future__ import annotations # 避免类型在定义顺序不对时出错 让类型注解（type hints）延迟解析（变成字符串）

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from augmentations import AstroMultiCropAugmentation
from dataset import build_dataset
from masking import collate_data_and_cast
from param_groups import build_head_param_groups, build_param_groups
from ssl_meta_arch import SSLMetaArch
from utils import (
    AverageMeter,
    CosineScheduler,
    compute_scaled_lr,
    ensure_dir,
    load_config,
    set_seed,
)

 # TO DO 补充导入包
import csv
import numpy as np
import matplotlib.pyplot as plt
from augmentations import AstroEvalTransform
import torch.nn.functional as F

#! Use
 # TO DO change i m port; get_args;  save_checkpoint; 
 # TO DO add load_checkpoint, save_history_json, plot_loss_curves, evaluate_ssl_loss, evaluate_representation_gap, export_test_embeddings_and_visuals, export_learnable_parameters
 # TO DO 在main函数中加入两个用于测试的DataLoader 加入resume逻辑和history初始化 更新整个训练循环 改成可记录历史 按 epoch 做测试

def get_args():
    parser = argparse.ArgumentParser() # 解析你在终端输入的参数
    parser.add_argument("--config", type=str, default="config.yaml") # 参数（文件）类型是字符串 #! .sh文件操作 “config_continue.yaml"
    parser.add_argument("--resume", type=str, default="")  # TO DO Resume option
    return parser.parse_args()


def build_schedulers(cfg, effective_lr):
    epoch_len = cfg["train"]["OFFICIAL_EPOCH_LENGTH"]
    total_iters = cfg["optim"]["epochs"] * epoch_len
    warmup_iters = cfg["optim"]["warmup_epochs"] * epoch_len

    return {
        "lr": CosineScheduler( 
            # 用余弦函数控制“某个参数”随训练逐步变化
            # 参数： 初始值 base_value 最终值 final_value 总步数T 当前步t 
            # 时间增加 参数平滑减小(前期快 后期平缓 适合训练后期精细调整) 最后T时为final_value
            base_value=effective_lr,
            final_value=cfg["optim"]["min_lr"],
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            start_warmup_value=0.0, # warmup阶段一开始的初始值 
            # 完整调度为 start_warmup_value → base_value → final_value  
            # warmup线性上升到base_value(平稳启动) 然后cos平滑下降
        ),
        "wd": CosineScheduler( # 直接cosine 没有warmup
            base_value=cfg["optim"]["weight_decay"],
            final_value=cfg["optim"]["weight_decay_end"],
            total_iters=total_iters,
        ),
        "momentum": CosineScheduler(
            base_value=cfg["teacher"]["momentum_teacher"],
            final_value=cfg["teacher"]["final_momentum_teacher"],
            total_iters=total_iters, 
            # 每一步(iteration)更新（非常平滑） “连续动态稳定器” 
            # 要每步都微调一点点 否则 teacher突然变化 → student目标乱跳 → 训练不稳定 
        ),
        "teacher_temp": CosineScheduler( # 控制输出分布的“尖锐程度” 不必精细变化
            base_value=cfg["teacher"]["teacher_temp"],
            final_value=cfg["teacher"]["teacher_temp"],
            total_iters=cfg["teacher"]["warmup_teacher_temp_epochs"] * epoch_len, 
            # 按照iteration从开始值不断warmup到warmup iteration结束 一直保持temp值 
            #于是不必指定total_iters 因为只要在代码中说明之后的temp值都和最后一个warmup值一样即可
            # 总之 teacher_temp 的 schedule 只覆盖 warmup 阶段 之后靠“截断索引”保持最后一个值
            warmup_iters=cfg["teacher"]["warmup_teacher_temp_epochs"] * epoch_len,
            start_warmup_value=cfg["teacher"]["warmup_teacher_temp"],
        ),
        "last_layer_lr": CosineScheduler(
            base_value=effective_lr,
            final_value=cfg["optim"]["min_lr"],
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            start_warmup_value=0.0,
        ),
    }


def apply_schedules(optimizer, lr, wd, last_layer_lr):
    for group in optimizer.param_groups: 
        # param_groups是一个列表 每一个group是一个字典 
        # 其中包括“params”,"lr","weight_decay"，“lr_scale”，“is_last_layer”等键及其值
        group["weight_decay"] = wd if group["weight_decay"] > 0 else 0.0
        # 只有原本有 weight_decay 的组才更新 有些参数（比如 bias / norm） 不应该加 weight decay
        current_lr = last_layer_lr if group.get("is_last_layer", False) else lr
        # group.get("key", default) 如果 key 存在 → 返回它的值 如果不存在 → 返回 default
        # is_last_layer 不是 PyTorch 自带的 是你们代码在构建 optimizer 时手动加进去的
        # 单独处理last layer的原因： 在DINO/SSL 里 最后一层最容易 collapse
        group["lr"] = current_lr * group.get("lr_scale", 1.0) 
        # layer-wise learning rate decay 最终学习率 = 基础lr × 层级缩放


def save_checkpoint(path, model, optimizer, iteration, epoch, history, cfg):  # TO DO add inputs: epoch, history, cfg
    payload = {
        "model": model.student["backbone"].state_dict(), 
        # "model" = student 的 backbone（只包含ViT编码器） 目标： 下游任务或推断-> 特征提取 embedding finetune
        "student": model.student.state_dict(),
        # "student" = 完整 student 网络（backbone + projection head (用于训练)） 目标：恢复预训练 重算loss
        "teacher": model.teacher.state_dict(),
        # "teacher" = 完整 teacher 网络（backbone + projection head） EMA累计结果 不能直接由student得出
        "optimizer": optimizer.state_dict(),
        "iteration": iteration, # 保存当前训练到第几步 用于resume training (断点续训)

         # TO DO add following items to be saved
        "epoch": epoch,
        "history": history,
        "cfg": cfg,
    }
    torch.save(payload, path)

 # TO DO add function load_checkpoint  Continual Training
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    if "student" in ckpt:
        model.student.load_state_dict(ckpt["student"], strict=True)
    elif "model" in ckpt:
        # fallback: only backbone was saved
        model.student["backbone"].load_state_dict(ckpt["model"], strict=False)

    if "teacher" in ckpt:
        model.teacher.load_state_dict(ckpt["teacher"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    start_iteration = int(ckpt.get("iteration", 0))
    start_epoch = int(ckpt.get("epoch", 0))
    history = ckpt.get("history", {
        "train_iter_loss": [],
        "train_epoch_loss": [],
        "test_ssl_loss": [],
    })
    return start_iteration, start_epoch, history

 # TO DO newly add
def save_history_json(path, history):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2) # json.dump写到文件 把数据history写入文件f（格式序列化成 JSON）

 # TO DO newly add
def plot_loss_curves(history, out_dir):
    out_dir = Path(out_dir)

    # iteration-level train loss
    if len(history["train_iter_loss"]) > 0:
        xs = [x["iteration"] for x in history["train_iter_loss"]]
        ys = [x["loss"] for x in history["train_iter_loss"]]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys)
        plt.xlabel("Iteration")
        plt.ylabel("Train SSL Loss")
        plt.title("Training Loss Curve")
        plt.tight_layout()
        plt.savefig(out_dir / "train_iter_loss.png", dpi=150) # dpi Dots Per Inch（每英寸像素数） 控制你保存图片的清晰度 / 分辨率
        plt.close()

    # epoch-level train / test
    if (
        len(history["train_epoch_loss"]) > 0
        or len(history["test_ssl_loss"]) > 0
        or len(history["test_cosine_gap"]) > 0
    ):
        plt.figure(figsize=(8, 5))
        if len(history["train_epoch_loss"]) > 0:
            xs = [x["epoch"] for x in history["train_epoch_loss"]]
            ys = [x["loss"] for x in history["train_epoch_loss"]]
            plt.plot(xs, ys, label="train_epoch_ssl_loss")
        if len(history["test_ssl_loss"]) > 0:
            xs = [x["epoch"] for x in history["test_ssl_loss"]]
            ys = [x["loss"] for x in history["test_ssl_loss"]]
            plt.plot(xs, ys, label="test_ssl_loss")
        if len(history["test_cosine_gap"]) > 0:
            xs = [x["epoch"] for x in history["test_cosine_gap"]]
            ys = [x["value"] for x in history["test_cosine_gap"]]
            plt.plot(xs, ys, label="test_cosine_gap")

        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Epoch Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "epoch_metrics.png", dpi=150) #! 注意这些图像在每次断点续训之后会覆盖之前的图像（图像名字一样） 但是之前的那些iteration的信息都是保留的 依然会画进图里 通过改out_dir解决了
        plt.close()

 # TO DO add held-out test loss evaluation function, evaluate_representation_gap
# 直接在 test split 上跑和训练一样的 forward_backward(...)，但不做 optimizer.step() 
# 它本质上是“测试集自监督损失”，不是分类 accuracy, 当前代码体系里就只有这个损失定义
def evaluate_ssl_loss(model, test_loader, teacher_temp, max_batches=None):
    # 保存模型原来的状态，并在函数结束后恢复
    was_training = model.training # 布尔值
    model.train()  # teacher 仍会保持 eval，因为 SSLMetaArch.train() 里已处理
    '''
    这个“测试”不是普通 inference 而是 self-supervised loss(DINO / iBOT loss)
    SSL loss 必须用 train 模式 因为 DINO / AstroCLIP 里面 关键机制依赖 train 模式：
    BatchNorm(如果有) Dropout (train时开启)

    model.eval() student变成eval dropout关闭 loss行为变化  SSL loss不对
    '''

    meter = AverageMeter()
    
    for step, batch in enumerate(test_loader): #! 这里的batch是经过crop的图像吗？ 必须是 否则有问题
        # 一次 SSL loss 计算 = 用整个 batch, 用该batch里每个样本生成出来的 所有 views（crops）一起算
        if max_batches is not None and step >= max_batches:
            break

        model.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(batch, teacher_temp=teacher_temp)
        loss_val = sum(v.item() for v in loss_dict.values())
        meter.update(loss_val)

        # 只计算，不更新参数
        model.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    return meter.avg

@torch.no_grad()
def evaluate_representation_gap(model, test_loader_embed, max_batches=10):
    device = next(model.parameters()).device
    student_backbone = model.student["backbone"]
    teacher_backbone = model.teacher["backbone"]

    prev_mode = model.training
    model.eval()

    meter = AverageMeter()
    n_done = 0

    for batch in test_loader_embed:
        if n_done >= max_batches: # 控制只评估前 N 个 batch 防止 test 太慢
            break

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device, non_blocking=True)

        s = student_backbone(x, is_training=True)["x_norm_clstoken"]
        t = teacher_backbone(x, is_training=True)["x_norm_clstoken"]

        s = F.normalize(s, dim=-1) # s.shape = (B, D) --> 对每个 embedding 做 L2 归一化之后 (B,D) 
        t = F.normalize(t, dim=-1) # t.shape = (B, D)

        cosine = (s * t).sum(dim=-1).mean() # dot product is just cosine similarity 
        # s*t是 逐元素乘： (B, D) * (B, D) → (B, D)  sum得到每个样本的点积：(B, D) → (B,)  
        # 每个元素就是：cosine similarity between s_i and t_i 取平均值
        gap = 1.0 - cosine.item() # cosine= -->完全一致 gap小

        meter.update(gap, n=x.shape[0]) # gap 是 batch 平均值 内部 sum += gap * n  count += n  avg = sum / count
        n_done += 1

    if prev_mode:
        model.train()
    else:
        model.eval()

    return meter.avg

 # TO DO newly add  export_test_embeddings_and_visuals    final embedding! 
@torch.no_grad()
def export_test_embeddings_and_visuals(model, test_embed_loader, out_dir, max_samples=256):
    out_dir = Path(out_dir)
    emb_dir = out_dir / "embeddings"
    ensure_dir(emb_dir)

    device = next(model.parameters()).device
    backbone = model.teacher["backbone"]  # teacher 更稳定，适合导出最终表征
    backbone.eval()

    all_embs = []
    all_imgs = []

    for batch in test_embed_loader:
        # 不管 batch 长什么样，都只取“图像 tensor”
        if isinstance(batch, (list, tuple)): # dataset 返回 (image, label)
            x = batch[0] # x = batch[0] 只取图像
        else: # dataset只返回图像的情况  batch = tensor[B, C, H, W]
            x = batch
        x = x.to(device, non_blocking=True)

        feats = backbone(x, is_training=True) # 把图像变成特征（embedding）feats是一个dict 不是普通tensor
        # is_training=True 保持和训练一致的表示分布
        cls = feats["x_norm_clstoken"]  # (B, D) 取出 CLS token（整张图的表示）

        all_embs.append(cls.cpu()) # 把数据从 GPU 移到 CPU，避免占用显存，并方便后续处理（保存 / 可视化 / numpy 等）
        all_imgs.append(x.cpu())

        total = sum(t.shape[0] for t in all_embs) 
        # 限制最多处理多少样本，防止爆内存 / 时间过长
        # 统计目前已经收集了多少 embedding (样本)，如果超过上限就停止循环
        '''
        all_embs = [
            tensor[B1, D],
            tensor[B2, D],
            tensor[B3, D],
            ...
        ]
        t.shape[0] 表示这个 batch 有多少张图(多少个 embedding)
        '''
        if total >= max_samples: # 最后一个 batch 可能会超过 max_samples 一点点 下面用[:max_samples] 考虑到了
            break

    embs = torch.cat(all_embs, dim=0)[:max_samples]   #  (B1+B2+...+B_max, D) 注意all_embs是list 所以会在第 0 维（batch 维）拼接
    imgs = torch.cat(all_imgs, dim=0)[:max_samples]   # (B1+B2+...+B_max, C, H, W)
    '''
    all_imgs = [
    tensor[B1, C, H, W],
    tensor[B2, C, H, W],
    ...
    ]
    '''

    torch.save(
        {
            "cls_embeddings": embs,
            "images": imgs,
        },
        emb_dir / "test_cls_embeddings.pt",
    )

    #! 在测试集上检查 embedding 有没有塌缩 分散 分群 异常样本 尺度问题
    # 1) embedding norm histogram 看1.表征尺度 不能几乎一样（可能collapse）2. 异常样本 3. 比较不同 checkpoint 的 norm histogram--训练稳定性
    norms = embs.norm(dim=1).numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(norms, bins=30)
    plt.xlabel("L2 norm")
    plt.ylabel("Count")
    plt.title("Test CLS Embedding Norm Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "test_embedding_norm_hist.png", dpi=150)
    plt.close()

    # 2) cosine similarity heatmap 看有没有塌缩  如果热图几乎整片都很亮，说明任意两张图都很相似
    # 健康的情况通常是 对角线最亮（自己和自己最像）  非对角元素有亮有暗，不是整片同色
    # 数据中是否有簇结构 如果热图中出现一块一块的亮方块，说明某些样本彼此更相似，可能形成了 cluster。 可能是相似天体类型
    embs_norm = torch.nn.functional.normalize(embs, dim=1) # 把每个 embedding 向量变成“单位长度”（L2 归一化）
    sim = embs_norm @ embs_norm.t()
    sim_np = sim.numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(sim_np, aspect="auto")
    plt.colorbar()
    plt.title("Test CLS Cosine Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "test_cosine_similarity_heatmap.png", dpi=150)
    plt.close()

    # 3) PCA 2D 看是否整体塌成一团（如果是的话 说明高维表示差异不大-> collapse） 
    # 是否有明显分群（出现多个团块，说明模型学到了一些可分结构）
    # 不是分成离散几团，而是沿某条弯曲带状分布，可能说明模型学到的是连续属性
    # 不同 checkpoint 的表征演化   比较训练前后 不同 epoch 的 PCA （早期可能杂乱 后期可能出现更清晰结构）
    x = embs - embs.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(x, q=2) # PCA（主成分分析）降维，(PyTorch 提供的高效低秩版本)
    # 把高维数据 x 降到 2 维（q=2），用于可视化或分析
    proj = x @ V[:, :2]
    proj = proj.numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Test CLS Embeddings PCA")
    plt.tight_layout()
    plt.savefig(out_dir / "test_embedding_pca.png", dpi=150)
    plt.close()
    # 看到奇怪现象时，最好做“图点回看” （Experience）

 # TO DO  Newly add export_layer_embeddings
#导出每层 embedding并分析
@torch.no_grad()
def export_layer_embeddings(model, test_embed_loader, out_dir, max_samples=128):
    out_dir = Path(out_dir)
    layer_dir = out_dir / "layer_embeddings"
    ensure_dir(layer_dir)

    device = next(model.parameters()).device
    backbone = model.teacher["backbone"]
    backbone.eval()

    collected_x = []

    for batch in test_embed_loader: # DataLoader处理后的测试集数据 （不是embedding 是用来提取embedding的）
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        collected_x.append(x)
        total = sum(t.shape[0] for t in collected_x)
        if total >= max_samples:
            break

    x = torch.cat(collected_x, dim=0)[:max_samples].to(device)

    n_blocks = len(backbone.blocks)
    outputs = backbone.get_intermediate_layers( # 运行得到每层表示
        x,
        n=list(range(n_blocks)),
        reshape=False,
        return_class_token=True,
        norm=True,
    )# 结果是一个 list，每个元素是某几层 Transformer 的输出 tuple(zip(outputs, class_tokens)) 

    save_obj = {}
    for layer_idx, (patch_tokens, cls_tokens) in enumerate(outputs):
        # patch_tokens: (B, N, D), cls_tokens: (B, D)
        patch_mean = patch_tokens.mean(dim=1) # （B,D）
        save_obj[f"layer_{layer_idx}_cls"] = cls_tokens.cpu()
        save_obj[f"layer_{layer_idx}_patch_mean"] = patch_mean.cpu()

    torch.save(save_obj, layer_dir / "all_layers_embeddings.pt")
    # 为了别把文件炸得太大 先导出每层的 CLS embedding 以及 每层 patch token 的平均向量 patch_mean_embedding

 # TO DO newly add export_learnable_parameters
# 导出全部可学习参数和统计摘要 
# 保存一个 parameters_full.pt：完整参数张量 再保存一个 parameters_summary.csv：每个参数的均值/标准差/最小/最大/是否有 NaN
def export_learnable_parameters(model, out_dir):
    out_dir = Path(out_dir)
    param_dir = out_dir / "parameters"
    ensure_dir(param_dir)

    full_state = {}
    rows = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        t = param.detach().cpu()
        full_state[name] = t

        rows.append({
            "name": name,
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()) if t.numel() > 1 else 0.0,
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "has_nan": bool(torch.isnan(t).any().item()), # 检查 tensor t 里面有没有 NaN（非法数值）
            # torch.isnan(t) 返回一个 布尔 tensor  .any() 只要有一个 True → 返回 True ... .item()->把 tensor 转成 Python 标量
            "has_inf": bool(torch.isinf(t).any().item()),
        })

    torch.save(full_state, param_dir / "parameters_full.pt") #! 注意这个也会覆盖之前的参数 最好的方法是直接把out_dir整个换掉

    with open(param_dir / "parameters_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "shape", "numel", "mean", "std", "min", "max", "has_nan", "has_inf"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args() # 从终端读取你输入的参数，并整理成一个对象 
    # 终端输入python train.py --config config.yaml  
    # args = get_args() 解析 "--config config.yaml"  得到args.config == "config.yaml"
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is used.")
    out_dir = Path(cfg["train"]["output_dir"]) #! 断点续训需要在config文件中替换
    ensure_dir(out_dir) # 如果 ./outputs 不存在 → 创建 如果已经存在 → 什么也不做

    transform = AstroMultiCropAugmentation(
        global_crops_scale=cfg["crops"]["global_crops_scale"],
        local_crops_scale=cfg["crops"]["local_crops_scale"],
        local_crops_number=cfg["crops"]["local_crops_number"],
        global_crops_size=cfg["crops"]["global_crops_size"],
        local_crops_size=cfg["crops"]["local_crops_size"],
        blur_probability=cfg["astro"]["blur_probability"],
        noise_probability=cfg["astro"]["noise_probability"],
        use_astro_augmentations=cfg["astro"]["use_astro_augmentations"],
    )
    dataset = build_dataset(cfg, transform=transform)

    num_patches = (cfg["crops"]["global_crops_size"] // cfg["student"]["patch_size"]) ** 2

    #!补充generator随机性部分 把 DataLoader 的随机数生成器“锁死” #有副作用 导致每次训练用的数据都一样
    #g = torch.Generator()
    #g.manual_seed(cfg["train"]["seed"])

    #! 补充work_init_fn
    def worker_init_fn(worker_id): #DataLoader 在启动每个 worker 进程时自动调用它 给出worker_id
        worker_seed = torch.initial_seed() % 2**32 # 用 generator 生成一个 base_seed → 分配给每个 worker
        set_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True, # 数据流：硬盘 -> CPU -> GPU  但CPU → GPU 传输速度可能成为瓶颈
        # pin_memory 把数据放在“锁页内存”（pinned memory）里 GPU可以直接DMA传输（更快）
        # 帮助支持CPU->GPU异步拷贝 更高效
        drop_last=True,
        collate_fn=lambda batch: collate_data_and_cast(batch, cfg, num_patches),
        #! DataLoader 默认batch = [sample1, sample2, ...]→ stack 成 tensor
        # 但该项目 一个样本 = 多个crop（global + local） 要自定义拼接方式
        # batch 一个 list，里面是 dataset 返回的多个样本
        # generator = g, #! 补充 又删除 因为g也设置随机种子 每次shuffle的不同batch排列顺序会相同 那么每次只用同样的前epoch_len 个batch来训练！
        worker_init_fn=worker_init_fn, #! 补充
    ) #! 补充一个generator的随机种子和不同worker的随机种子
    # DataLoader也有随机性 DataLoader 有自己的随机系统，和全局 seed 是“分开的”
    # DataLoader worker 是独立进程 不共享主进程的 seed

     # TO DO 加入用于测试的2个loader SSLloss测试+推断/representation gap
    # test loader for held-out SSL loss (same augmentation style as training)
    test_dataset_ssl = build_dataset(cfg, transform=transform, split="test")
    test_loader_ssl = DataLoader(
        test_dataset_ssl,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_data_and_cast(batch, cfg, num_patches),
        worker_init_fn=worker_init_fn,
    )

    # deterministic loader for embedding export / visualization
    eval_transform = AstroEvalTransform(
        image_size=cfg["crops"]["global_crops_size"],
        use_astro_augmentations=cfg["astro"]["use_astro_augmentations"],
    )
    test_dataset_embed = build_dataset(cfg, transform=eval_transform, split="test")
    # embedding = 模型对数据的映射 换数据 → embedding 会变
    # 之后的CLIP 训练的是“如何生成 embedding”，而不是 embedding 本身
    test_loader_embed = DataLoader(
        test_dataset_embed,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )


    total_batch_size = cfg["train"]["batch_size_per_gpu"]
    effective_lr = compute_scaled_lr(
        cfg["optim"]["base_lr"],
        total_batch_size,
        cfg["optim"]["scaling_rule"],
    )
    cfg["optim"]["lr"] = effective_lr # 修改config文件 让整个系统“统一使用更新后的lr”

    model = SSLMetaArch(cfg).to(device)  # 配置文件不变 断点续训 模型不会变化
    model.train()

    param_groups = [] # 构建“参数分组”，然后加入到 param_groups 里
    param_groups.extend( # extend = 把“多个参数组” get from a list of groups 再逐个加入列表 [group1, group2,...]
        build_param_groups( 
            model.student["backbone"],
            base_lr=effective_lr,
            weight_decay=cfg["optim"]["weight_decay"],
            layerwise_decay=cfg["optim"]["layerwise_decay"],
            patch_embed_lr_mult=cfg["optim"]["patch_embed_lr_mult"],
        )
    )
    param_groups.extend(
        build_head_param_groups(
            model.student["dino_head"],
            base_lr=effective_lr,
            weight_decay=cfg["optim"]["weight_decay"],
            is_last_layer=True, # 手动写入参数组新的键
            # DINO只关心“整张图的语义表示” 所以只需要在最后一层（最强语义表达）
            # DINO： ViT → CLS → head → loss
        )
    )
    if "ibot_head" in model.student:
        param_groups.extend(
            build_head_param_groups(
                model.student["ibot_head"],
                base_lr=effective_lr,
                weight_decay=cfg["optim"]["weight_decay"],
                is_last_layer=False, 
                # 既学习 global 表示 又学习 token-level 表示（类似 masked modeling）
                # 不是只在最后看CLS 还要对每个token进行预测
                # ViT → tokens → head → loss
                #      ↘ CLS → head → loss（有时也保留）
            )
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg["optim"]["adamw_beta1"], cfg["optim"]["adamw_beta2"]),
    )

    sched = build_schedulers(cfg, effective_lr)
    freeze_last_layer_iters = cfg["optim"]["freeze_last_layer_epochs"] * cfg["train"]["OFFICIAL_EPOCH_LENGTH"]

     # TO DO 加入resume逻辑和history初始化
    history = {
    "train_iter_loss": [],
    "train_epoch_loss": [],
    "test_ssl_loss": [],
    "test_cosine_gap": [],
    }

    iteration = 0
    start_epoch = 0

    resume_path = args.resume if args.resume else cfg["train"].get("resume_path", "") #! 断点续训需要修改resume_path 从""变成真实resume path
    if resume_path: # 这样wall time 不够也不会白跑 #  resume_path 是字符串
        iteration, start_epoch, history = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer, # 优化器参数断点续训
            map_location=device,
        )
        print(f"Resumed from {resume_path} | start_epoch={start_epoch}, iteration={iteration}")

     # TO DO 更新整个训练循环 改成可记录历史 按 epoch 做测试
     # TO DO loss_meter分为2个 
    epoch_len = cfg["train"]["OFFICIAL_EPOCH_LENGTH"]
    global_loss_meter = AverageMeter() # 全局平均 

    for epoch in range(start_epoch, cfg["optim"]["epochs"]): #! 用于wall time不够时的断点续训 但是如果已经完成了所有epoch的训练 还想再训练可以直接去配置文件改epoch数 表示计划达到的总epoch数
    #如果 for epoch in range(start_epoch, start_epoch + cfg["optim"]["epochs"]): 前一次全部epoch完成 这次额外再训练cfg["optim"]["epochs"]   可能和一些函数语义不一致
        epoch_loss_meter = AverageMeter() # 每个 epoch 的损失

        for step, batch in enumerate(loader): # loader是训练数据集
            if step >= epoch_len: 
                #! 注意在取10%数据的情况下 epoch_len (1250)>len(loader) (17000/72~=236) 所以所有的数据在每个epoch中一定会被用到
                #! 如果用全部数据 epoch_len (1250)<len(loader) (170000/72~=2360) 一个epoch中会有些数据没有被用到 所有epoch下来还有可能有batch没有被用过
                break

            lr = sched["lr"][iteration] #! 注意这里有iteration所以断点续训优化器参数也是接着之前的训练继续变化的 没有跳跃
            wd = sched["wd"][iteration]
            last_layer_lr = 0.0 if iteration < freeze_last_layer_iters else sched["last_layer_lr"][iteration]
            teacher_temp = sched["teacher_temp"][min(iteration, len(sched["teacher_temp"].schedule) - 1)]
            momentum = sched["momentum"][iteration]

            apply_schedules(optimizer, lr, wd, last_layer_lr)

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.forward_backward(batch, teacher_temp=teacher_temp)

            if cfg["optim"]["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg["optim"]["clip_grad"])

            optimizer.step()
            model.update_teacher(momentum)

            loss_val = sum(v.item() for v in loss_dict.values())
            global_loss_meter.update(loss_val)
            epoch_loss_meter.update(loss_val)

             # TO DO 增加 history
            history["train_iter_loss"].append({ # 不会覆盖断点之前的history会在之前的基础上一直补充
                "iteration": int(iteration), # iteration即step
                "epoch": int(epoch),
                "loss": float(loss_val),
                "lr": float(lr),
                "wd": float(wd),
                "teacher_temp": float(teacher_temp),
                "momentum": float(momentum),
            })

            if iteration % 20 == 0:
                pretty = {k: round(v.item(), 4) for k, v in loss_dict.items()}
                print(
                    f"iter={iteration:05d} epoch={epoch:03d} "
                    f"lr={lr:.6f} wd={wd:.4f} temp={teacher_temp:.4f} "
                    f"mom={momentum:.5f} loss={loss_val:.4f} details={pretty}"
                )

            if iteration > 0 and iteration % (cfg["train"]["saveckp_freq"] * epoch_len) == 0:
                save_checkpoint(
                    out_dir / f"checkpoint_{iteration}.pt",
                    model,
                    optimizer,
                    iteration,
                    epoch,
                    history,
                    cfg,
                )
                save_history_json(out_dir / "history.json", history)
                plot_loss_curves(history, out_dir)

            iteration += 1

        # epoch end
        history["train_epoch_loss"].append({
            "epoch": int(epoch),
            "loss": float(epoch_loss_meter.avg),
        })

        # held-out SSL loss on test split 
        test_ssl = evaluate_ssl_loss(
            model,
            test_loader_ssl,
            teacher_temp=teacher_temp,
            max_batches=epoch_len,
        )
        history["test_ssl_loss"].append({
            "epoch": int(epoch),
            "loss": float(test_ssl),
        })

        # 注意 SSL可能很小 但是模型collapse 所以还需要另一个评估指标表示模型实际的“表示能力”
        # -> test_gap = evaluate_representation_gap(...)
        test_gap = evaluate_representation_gap(
        model,
        test_loader_embed,
        max_batches=cfg["evaluation"]["test_eval_max_batches"],
        )
        history["test_cosine_gap"].append({
            "epoch": int(epoch),
            "value": float(test_gap),
        })

        print(
            f"[epoch {epoch}]" 
            f"train_epoch_ssl_loss={epoch_loss_meter.avg:.4f} "
            f"test_ssl_loss={test_ssl:.4f}"
            f"test_cosine_gap={test_gap:.6f}"  
        )

         # TO DO 自动保存checkpoint + history
        save_history_json(out_dir / "history.json", history)
        plot_loss_curves(history, out_dir)

        # save epoch checkpoint
        save_checkpoint(
            out_dir / f"epoch_{epoch:03d}.pt", # 断点续训后checkpoint的epoch应该是无缝衔接的
            model,
            optimizer,
            iteration,
            epoch + 1,
            history,
            cfg,
        )

    # final save
    save_checkpoint(
        out_dir / "final_checkpoint.pt",
        model,
        optimizer,
        iteration, # iteration 是从头到尾总共的迭代step数
        epoch+1, #! 原先代码 cfg["optim"]["epochs"], 不能说明断点续训前最后一次走了多少个epoch 默认训练完全部epoch了 这个参数是下次断点续训应该用的起始epoch序数
        history,
        cfg,
    )

    # final exports
    export_test_embeddings_and_visuals(model, test_loader_embed, out_dir, max_samples=cfg["evaluation"]["export_max_samples"])
    export_layer_embeddings(model, test_loader_embed, out_dir, max_samples=cfg["evaluation"]["export_layer_max_samples"])
    export_learnable_parameters(model, out_dir)

    print(f"Training finished. Average loss={global_loss_meter.avg:.4f}")


if __name__ == "__main__": 
# 只有直接运行这个文件时 （设置 __name__ = "__main__"），才执行 main() 
#否则import时 （设置 __name__ = "train"）也会自动执行训练
    main()

# __name__ = 当前模块的名字
# "__main__" = 直接运行的入口文件

    '''
    原来的训练循环部分
    iteration = 0
    loss_meter = AverageMeter() # 用来实时统计“平均loss”的工具 帮助记录 当前值+累计和+平均值(单步loss波动很大)
    # 把当前这个 step 的 loss 记录进去，并更新“平均 loss”
    epoch_len = cfg["train"]["OFFICIAL_EPOCH_LENGTH"]

    for epoch in range(cfg["optim"]["epochs"]):
        for step, batch in enumerate(loader):
            if step >= epoch_len:
                break

            lr = sched["lr"][iteration] # sched["lr"] 一个“列表/数组”（提前算好的学习率序列) [iteration]第 iteration 步应该用的学习率
            wd = sched["wd"][iteration]
            last_layer_lr = 0.0 if iteration < freeze_last_layer_iters else sched["last_layer_lr"][iteration]
            teacher_temp = sched["teacher_temp"][min(iteration, len(sched["teacher_temp"].schedule) - 1)] 
            # utils 里的 CosineScheduler 返回一个对象 可以用__getitem__来实现像数组一样索引 有属性.schedule可以访问整个数组
            # 如果超过 就用最后一个值
            momentum = sched["momentum"][iteration]

            apply_schedules(optimizer, lr, wd, last_layer_lr)

            optimizer.zero_grad(set_to_none=True) # set_to_none=True = 把梯度设为 None，而不是设为 0 (by default)
            # Pros: 更省内存 更快 更符合PyTroch的底层设计 --》 oaram.grad = None
            loss_dict = model.forward_backward(batch, teacher_temp=teacher_temp)

            if cfg["optim"]["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg["optim"]["clip_grad"])
            optimizer.step()
            model.update_teacher(momentum)

            loss_val = sum(v.item() for v in loss_dict.values())
            loss_meter.update(loss_val)

            if iteration % 20 == 0: #! 可能会有点频繁
                pretty = {k: round(v.item(), 4) for k, v in loss_dict.items()}
                print(
                    f"iter={iteration:05d} epoch={epoch:03d} "
                    f"lr={lr:.6f} wd={wd:.4f} temp={teacher_temp:.4f} "
                    f"mom={momentum:.5f} loss={loss_val:.4f} details={pretty}"
                )

            if iteration > 0 and iteration % (cfg["train"]["saveckp_freq"] * epoch_len) == 0:# 按epoch来存储模型
                save_checkpoint(out_dir / f"checkpoint_{iteration}.pt", model, optimizer, iteration)

            iteration += 1

    save_checkpoint(out_dir / "final_checkpoint.pt", model, optimizer, iteration)
    print(f"Training finished. Average loss={loss_meter.avg:.4f}")
    '''


