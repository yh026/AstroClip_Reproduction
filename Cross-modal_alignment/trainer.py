# #!/usr/bin/env python
# from typing import Any, Optional

# import matplotlib.pyplot as plt
# import wandb
# from lightning import Callback, LightningModule, Trainer
# from lightning.pytorch.cli import (
#     ArgsType,
#     LightningArgumentParser,
#     LightningCLI,
#     LRSchedulerTypeUnion,
# )
# from lightning.pytorch.loggers import WandbLogger
# from torch.optim import Optimizer

# from astroclip import format_with_env
# from astroclip.callbacks import CustomSaveConfigCallback


# class WrappedLightningCLI(LightningCLI):
#     def before_instantiate_classes(self) -> None:
#         self.config = format_with_env(self.config)

#     # Changing the lr_scheduler interval to step instead of epoch
#     @staticmethod
#     def configure_optimizers(
#         lightning_module: LightningModule,
#         optimizer: Optimizer,
#         lr_scheduler: Optional[LRSchedulerTypeUnion] = None,
#     ) -> Any:
#         optimizer_list, lr_scheduler_list = LightningCLI.configure_optimizers(
#             lightning_module, optimizer=optimizer, lr_scheduler=lr_scheduler
#         )

#         for idx in range(len(lr_scheduler_list)):
#             if not isinstance(lr_scheduler_list[idx], dict):
#                 lr_scheduler_list[idx] = {
#                     "scheduler": lr_scheduler_list[idx],
#                     "interval": "step",
#                 }
#         return optimizer_list, lr_scheduler_list


# def main_cli(args: ArgsType = None, run: bool = True):
#     cli = WrappedLightningCLI(
#         save_config_kwargs={"overwrite": True},
#         save_config_callback=CustomSaveConfigCallback,
#         parser_kwargs={"parser_mode": "omegaconf"},
#         args=args,
#         run=run,
#     )
#     return cli


# if __name__ == "__main__":
#     main_cli(run=True)

#=========================================================================================================================================
# train.py
# import torch
# from lightning import Trainer
# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.loggers import CSVLogger
# from torch.optim import AdamW

# from astroclip.models.astroclip import AstroClipModel, ImageHead, SpectrumHead
# from astroclip.data.datamodule import AstroClipDataloader
# from astroclip.scheduler import CosineAnnealingWithWarmupLR  # 假设你有这个文件

# # -----------------------------
# # 1️⃣ 初始化 datamodule
# # -----------------------------
# datamodule = AstroClipDataloader(
#     path="/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10",
#     batch_size=256,
#     num_workers=8,
#     center_crop=144
# )
# datamodule.setup()  # 提前加载数据集

# # -----------------------------
# # 2️⃣ 初始化模型
# # -----------------------------
# image_encoder = ImageHead(
#     config="astroclip/astrodino/config.yaml",
#     model_weights="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/astrodino.ckpt",
#     save_directory="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs/astrodino"
# )
# spectrum_encoder = SpectrumHead(
#     model_path="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/specformer.ckpt"
# )

# model = AstroClipModel(
#     image_encoder=image_encoder,
#     spectrum_encoder=spectrum_encoder
# )

# # -----------------------------
# # 3️⃣ 初始化优化器和 lr_scheduler
# # -----------------------------
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
# scheduler = CosineAnnealingWithWarmupLR(optimizer, T_max=10000, T_warmup=1000, eta_min=1e-4/500)

# # -----------------------------
# # 4️⃣ 初始化 callbacks / logger
# # -----------------------------
# checkpoint_callback = ModelCheckpoint(
#     save_last=True,
#     save_top_k=2,
#     every_n_epochs=1,
#     monitor="val_loss_nologit"
# )
# lr_monitor = LearningRateMonitor(logging_interval="step")
# logger = CSVLogger(save_dir="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs")

# # -----------------------------
# # 5️⃣ 初始化 Trainer
# # -----------------------------
# trainer = Trainer(
#     max_epochs=100,
#     enable_checkpointing=True,
#     gradient_clip_val=1.0,
#     detect_anomaly=True,
#     precision=32,
#     callbacks=[checkpoint_callback, lr_monitor],
#     logger=logger,
#     devices=1,
#     accelerator="gpu"
# )

# # -----------------------------
# # 6️⃣ 开始训练
# # -----------------------------
# trainer.fit(model, datamodule=datamodule)


# train_native.py
# import torch
# from astroclip.data.datamodule import AstroClipDataloader
# from astroclip.models.astroclip import AstroClipModel
# from astroclip.models.astroclip import ImageHead, SpectrumHead

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1️⃣ 数据加载
# dm = AstroClipDataloader()
# train_loader = dm.train_dataloader()
# val_loader = dm.val_dataloader()

# # 2️⃣ 初始化模型
# image_encoder = ImageHead(
#     config="astroclip/astrodino/config.yaml",
#     model_weights="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/astrodino.ckpt",
#     save_directory="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs/astrodino"
# )
# spectrum_encoder = SpectrumHead(
#     model_path="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/specformer.ckpt"
# )

# # model = AstroClipModel(image_encoder=image_encoder, spectrum_encoder=spectrum_encoder)
# # model.to(device)

# # for p in model.image_encoder.parameters():
# #     p.requires_grad = False

# # for p in model.spectrum_encoder.parameters():
# #     p.requires_grad = False

# # # 3️⃣ 优化器
# # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# # # 4️⃣ 训练循环
# # num_epochs = 100
# # for epoch in range(num_epochs):
# #     model.train()
# #     total_loss = 0
# #     for batch in train_loader:
# #         images = batch["image"].to(device)
# #         spectrum = batch["spectrum"].to(device)
# #         optimizer.zero_grad()
# #         loss = model.compute_loss(images, spectrum)
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item()
# #     print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {total_loss/len(train_loader):.6f}")

# #     # 5️⃣ 验证
# #     model.eval()
# #     val_loss = 0
# #     with torch.no_grad():
# #         for batch in val_loader:
# #             images = batch["image"].to(device)
# #             spectrum = batch["spectrum"].to(device)
# #             loss = model.compute_loss(images, spectrum)
# #             val_loss += loss.item()
# #     print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {val_loss/len(val_loader):.6f}")

# model = AstroClipModel(image_encoder=image_encoder, spectrum_encoder=spectrum_encoder)
# model.to(device)

# # ✅ freeze backbone
# for p in model.image_encoder.backbone.parameters():
#     p.requires_grad = False

# for p in model.spectrum_encoder.backbone.parameters():
#     p.requires_grad = False

# # ✅ optimizer
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=5e-4,
#     weight_decay=0.01
# )

# num_epochs = 100

# # ✅ scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=num_epochs
# )

# # ✅ early stopping
# best_val = float("inf")
# patience = 5
# wait = 0

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for batch in train_loader:
#         images = batch["image"].to(device)
#         spectrum = batch["spectrum"].to(device)

#         optimizer.zero_grad()
#         loss = model.compute_loss(images, spectrum)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.6f}")

#     # validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch["image"].to(device)
#             spectrum = batch["spectrum"].to(device)
#             loss = model.compute_loss(images, spectrum)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}")

#     # scheduler
#     scheduler.step()

#     # early stopping
#     if val_loss < best_val:
#         best_val = val_loss
#         wait = 0
#     else:
#         wait += 1
#         if wait >= patience:
#             print("Early stopping triggered")
#             break




#trainer.py
# import torch
# import os
# from astroclip.data.datamodule import AstroClipDataloader
# from astroclip.models.astroclip import AstroClipModel
# from astroclip.models.astroclip import ImageHead, SpectrumHead

# # --- 配置保存路径 ---
# # 确保这个目录存在，且你有写入权限。
# # 建议放在 /scratch 下，因为 /home 空间通常较小且IO较慢
# SAVE_DIR = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs1234/trained_models"
# os.makedirs(SAVE_DIR, exist_ok=True)
# MODEL_PATH = os.path.join(SAVE_DIR, "astroclip_best_model.pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1️⃣ 数据加载
# dm = AstroClipDataloader()
# train_loader = dm.train_dataloader()
# val_loader = dm.val_dataloader()

# # 2️⃣ 初始化模型
# image_encoder = ImageHead(
#     config="astroclip/astrodino/config.yaml",
#     model_weights="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/astrodino.ckpt",
#     save_directory="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs/astrodino"
# )
# spectrum_encoder = SpectrumHead(
#     model_path="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/specformer.ckpt"
# )

# model = AstroClipModel(image_encoder=image_encoder, spectrum_encoder=spectrum_encoder)
# model.to(device)

# # ✅ freeze backbone
# for p in model.image_encoder.backbone.parameters():
#     p.requires_grad = False

# for p in model.spectrum_encoder.backbone.parameters():
#     p.requires_grad = False

# # ✅ optimizer
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=5e-4,
#     weight_decay=0.01
# )

# num_epochs = 100

# # ✅ scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=num_epochs
# )

# # ✅ early stopping
# best_val = float("inf")
# patience = 5
# wait = 0

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for batch in train_loader:
#         images = batch["image"].to(device)
#         spectrum = batch["spectrum"].to(device)

#         optimizer.zero_grad()
#         loss = model.compute_loss(images, spectrum)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.6f}")

#     # validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch["image"].to(device)
#             spectrum = batch["spectrum"].to(device)
#             loss = model.compute_loss(images, spectrum)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}")

#     # scheduler
#     scheduler.step()

#     # early stopping & Saving Logic
#     if val_loss < best_val:
#         best_val = val_loss
#         wait = 0
#         # ✅ 保存最佳模型
#         print(f"✅ New best model found! Saving to {MODEL_PATH}")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_loss': val_loss,
#         }, MODEL_PATH)
#     else:
#         wait += 1
#         if wait >= patience:
#             print("Early stopping triggered")
#             break

# # ✅ 如果训练完整跑完没有触发 early stopping，也保存最后一次的状态
# if wait < patience:
#     print("Training finished. Saving final model...")
#     torch.save({
#         'epoch': num_epochs - 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_loss': val_loss,
#     }, MODEL_PATH.replace(".pt", "_final.pt"))

# print("=== Job Completed ===")





"""
trainer_official.py
使用官方预训练的 ImageHead / SpectrumHead 作为 backbone，
对齐层改为论文 Section 3.3 的 CrossAttentionHead + MLP。

与 trainer.py 的区别：
1. 对齐层从 Linear projection 改为 CrossAttentionHead（论文结构）
2. embedding 维度改为 512（论文）
3. logit_scale 固定为 15.5（论文 B3.2）
4. lr 改为 1e-4（论文值）
5. backbone 冻结方式不变
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 路径 ─────────────────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)

SAVE_DIR   = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "astroclip_official_best_new.pt")

# ── 导入 ──────────────────────────────────────────────────────────────────────
from astroclip.data.datamodule import AstroClipDataloader
from astroclip.models.astroclip import ImageHead, SpectrumHead


# ══════════════════════════════════════════════════════════════════════════════
# 论文 Section 3.3 / Appendix B3.1
# CrossAttentionHead：可学习 query → 多头交叉注意力 → 单向量
# ══════════════════════════════════════════════════════════════════════════════
class CrossAttentionHead(nn.Module):
    """
    用一个可学习 query 向量对 transformer 输出的 token 序列做多头交叉注意力，
    输出固定长度的单一向量 z ∈ R^embed_dim。
    论文参数：4 heads，embed_dim=512，后接 2 个 Linear + LayerNorm + GeLU
    """
    def __init__(self, embed_dim: int = 512, n_head: int = 4,
                 token_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习 query
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # key / value 投影
        self.kv_proj = nn.Linear(token_dim, embed_dim * 2, bias=False)

        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

        # 论文：LayerNorm + MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.xavier_uniform_(self.kv_proj.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, seq_len, token_dim)
        返回:   (B, embed_dim)
        """
        B    = tokens.size(0)
        kv   = self.kv_proj(tokens)
        k, v = kv.chunk(2, dim=-1)
        q    = self.query.expand(B, -1, -1)

        attn_out, _ = self.attn(q, k, v)
        attn_out    = attn_out.squeeze(1)

        x = self.norm1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# CLIP InfoNCE Loss，logit_scale 固定为 15.5（论文 B3.2）
# ══════════════════════════════════════════════════════════════════════════════
class CLIPLoss(nn.Module):
    def forward(self, img_feat: torch.Tensor, spec_feat: torch.Tensor,
                logit_scale: float = 15.5) -> torch.Tensor:
        img_feat  = F.normalize(img_feat,  dim=-1)
        spec_feat = F.normalize(spec_feat, dim=-1)

        logits_per_image    = logit_scale * img_feat @ spec_feat.T
        logits_per_spectrum = logits_per_image.T

        labels = torch.arange(img_feat.size(0), device=img_feat.device)
        loss_i = F.cross_entropy(logits_per_image,    labels)
        loss_s = F.cross_entropy(logits_per_spectrum, labels)
        return (loss_i + loss_s) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 官方 backbone 包装：输出全部 token 序列
# ══════════════════════════════════════════════════════════════════════════════
class OfficialImageEncoder(nn.Module):
    """
    包装官方 ImageHead，冻结 backbone，
    forward() 返回全部 patch token，shape (B, N, 1024)。
    """
    def __init__(self):
        super().__init__()
        self.head = ImageHead(
            config="astroclip/astrodino/config.yaml",
            model_weights=os.path.join(ASTROCLIP_ROOT, "pretrained/astrodino.ckpt"),
            save_directory=os.path.join(ASTROCLIP_ROOT, "outputs/astrodino"),
        )
        # 冻结 backbone
        for p in self.head.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: (B, N, 1024)  — 全部 patch token（经过 backbone norm）
        """
        with torch.no_grad():
            # 复用 ImageHead 里的 backbone 前向，取全部 token
            xp = self.head.backbone.patch_embed(x)
            for blk in self.head.backbone.blocks:
                xp = blk(xp)
            embedding = self.head.backbone.norm(xp)   # (B, N+1, 1024)
        # 去掉 CLS token，只返回 patch tokens
        return embedding[:, 1:, :]                    # (B, N, 1024)


class OfficialSpectrumEncoder(nn.Module):
    """
    包装官方 SpectrumHead，冻结 backbone，
    forward() 返回全部 token embedding，shape (B, seq_len, 768)。
    """
    def __init__(self):
        super().__init__()
        self.head = SpectrumHead(
            model_path=os.path.join(ASTROCLIP_ROOT, "pretrained/specformer.ckpt"),
        )
        # 冻结 backbone
        for p in self.head.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L)
        返回: (B, seq_len, 768)  — 全部 token embedding
        """
        with torch.no_grad():
            embedding = self.head.backbone(x)["embedding"]  # (B, seq_len, 768)
        return embedding


# ══════════════════════════════════════════════════════════════════════════════
# AstroCLIP 完整模型
# ══════════════════════════════════════════════════════════════════════════════
class AstroClipOfficial(nn.Module):
    """
    官方 backbone（冻结）+ 论文 CrossAttentionHead（可训练）。
    embedding 维度 512，logit_scale 固定 15.5。
    """
    IMG_TOKEN_DIM  = 1024
    SPEC_TOKEN_DIM = 768
    EMBED_DIM      = 512
    LOGIT_SCALE    = 15.5

    def __init__(self):
        super().__init__()
        self.image_encoder    = OfficialImageEncoder()
        self.spectrum_encoder = OfficialSpectrumEncoder()

        self.img_head  = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.IMG_TOKEN_DIM)
        self.spec_head = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.SPEC_TOKEN_DIM)

        self.criterion = CLIPLoss()

    def forward(self, images: torch.Tensor, spectrum: torch.Tensor):
        img_tokens  = self.image_encoder(images)      # (B, N, 1024)
        spec_tokens = self.spectrum_encoder(spectrum)  # (B, S, 768)

        img_feat  = self.img_head(img_tokens)          # (B, 512)
        spec_feat = self.spec_head(spec_tokens)        # (B, 512)
        return img_feat, spec_feat

    def compute_loss(self, images: torch.Tensor, spectrum: torch.Tensor):
        img_feat, spec_feat = self.forward(images, spectrum)
        return self.criterion(img_feat, spec_feat, self.LOGIT_SCALE)


# ══════════════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1️⃣ 数据加载
    dm           = AstroClipDataloader()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # 2️⃣ 构建模型
    model = AstroClipOfficial()
    model.to(device)

    # 3️⃣ 可训练参数统计（只有两个 CrossAttentionHead）
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Total params:     {total_params:,}")
    print(f"✅ Trainable params: {trainable_params:,}\n")

    # 4️⃣ Optimizer（只训练 CrossAttentionHead，论文 lr=1e-4）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )

    num_epochs = 100

    # 5️⃣ Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    # 6️⃣ Early stopping
    best_val = float("inf")
    patience = 5
    wait     = 0
    val_loss = float("inf")

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images   = batch["image"].to(device)
            spectrum = batch["spectrum"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(images, spectrum)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1:3d}] Train Loss: {total_loss / len(train_loader):.6f}")

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images   = batch["image"].to(device)
                spectrum = batch["spectrum"].to(device)
                val_loss += model.compute_loss(images, spectrum).item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1:3d}] Val   Loss: {val_loss:.6f}")

        scheduler.step()

        # ── Early stopping & checkpoint ──
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            print(f"✅ New best model! Saving to {MODEL_PATH}")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
            }, MODEL_PATH)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if wait < patience:
        final_path = MODEL_PATH.replace(".pt", "_final.pt")
        print(f"Training finished. Saving final model to {final_path}")
        torch.save({
            "epoch":                num_epochs - 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
        }, final_path)

    print("=== Job Completed ===")


if __name__ == "__main__":
    main()