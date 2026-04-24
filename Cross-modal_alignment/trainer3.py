#trainer3.py
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
# ── 路径 ─────────────────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)
 
SPECTRUM_CKPT = "/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt"
IMAGE_CKPT    = "/scratch/users/nus/e1554059/astroclip/image_encoder/Final_ViT_model/image_encoder_large_raw_epoch80.pt"
 
SAVE_DIR   = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "astroclip_trainer3_best.pt")
 
# ── 导入 ──────────────────────────────────────────────────────────────────────
from astroclip.data.datamodule import AstroClipDataloader
from astroclip.models.yibinspectrummodel import SpecFormer, SpecFormerConfig
from astroclip.models.largeimagemodel import vit_large
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 论文 Section 3.3 / Appendix B3.1
# CrossAttentionHead: 可学习 query → 多头交叉注意力 → 单向量
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
        self.n_head    = n_head
 
        # 可学习 query，shape (1, 1, embed_dim)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
 
        # key / value 投影（将 token_dim 投影到 embed_dim）
        self.kv_proj = nn.Linear(token_dim, embed_dim * 2, bias=False)
 
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,   # (B, seq, dim)
        )
 
        # 论文：交叉注意力后接 2 个 Linear + LayerNorm + GeLU
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
        B = tokens.size(0)
 
        # 投影 tokens 为 key 和 value
        kv    = self.kv_proj(tokens)                    # (B, seq, 2*embed_dim)
        k, v  = kv.chunk(2, dim=-1)                     # each (B, seq, embed_dim)
 
        # query 扩展到 batch
        q = self.query.expand(B, -1, -1)                # (B, 1, embed_dim)
 
        # 多头交叉注意力
        attn_out, _ = self.attn(q, k, v)                # (B, 1, embed_dim)
        attn_out = attn_out.squeeze(1)                  # (B, embed_dim)
 
        # 残差 + LayerNorm + MLP（参考论文 B3.1）
        x = self.norm1(attn_out)
        x = x + self.mlp(self.norm2(x))
 
        return x                                        # (B, embed_dim)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CLIP InfoNCE Loss（论文 Eq.1）
# logit_scale 固定为 15.5（论文 B3.2）
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
# 图像 Backbone：输出全部 token 序列
# ══════════════════════════════════════════════════════════════════════════════
class ImageBackboneWrapper(nn.Module):
    """
    vit_large backbone，冻结参数。
    forward() 返回全部归一化 token，shape (B, N+1, 1024)。
    （论文做法：把所有 token 送入交叉注意力头）
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
 
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        cfg   = ckpt["cfg"]["student"]
        crops = ckpt["cfg"].get("crops", {})
 
        img_size            = cfg.get("img_size", crops.get("global_crops_size", 224))
        patch_size          = cfg["patch_size"]
        in_chans            = cfg.get("in_chans", 3)
        drop_path_rate      = cfg.get("drop_path_rate", 0.0)
        drop_path_uniform   = cfg.get("drop_path_uniform", False)
        layerscale          = cfg.get("layerscale", None)
        num_register_tokens = cfg.get("num_register_tokens", 0)
 
        self.backbone = vit_large(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            init_values=layerscale,
            qkv_bias=cfg["qkv_bias"],
            proj_bias=cfg["proj_bias"],
            ffn_bias=cfg["ffn_bias"],
            num_register_tokens=num_register_tokens,
        )
        missing, unexpected = self.backbone.load_state_dict(ckpt["model"], strict=False)
        print(f"[ImageBackbone] missing={len(missing)}, unexpected={len(unexpected)}")
 
        # ✅ 冻结
        for p in self.backbone.parameters():
            p.requires_grad = False
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: (B, N+1, 1024)  — 全部归一化 token（CLS + patch tokens）
        """
        ret = self.backbone.forward_features(x)       # dict
        return ret["x_norm_patchtokens"]               # (B, N, 1024) patch tokens
        # 如需包含 CLS token 可改为：
        # cls   = ret["x_norm_clstoken"].unsqueeze(1)  # (B,1,1024)
        # patch = ret["x_norm_patchtokens"]            # (B,N,1024)
        # return torch.cat([cls, patch], dim=1)        # (B,N+1,1024)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 光谱 Backbone：输出全部 token 序列
# ══════════════════════════════════════════════════════════════════════════════
class SpectrumBackboneWrapper(nn.Module):
    """
    SpecFormer backbone，冻结参数。
    forward() 返回全部 token embedding，shape (B, seq_len, 768)。
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
 
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = {k: v for k, v in ckpt["config"].items()
                    if k in SpecFormerConfig.__annotations__}
        cfg = SpecFormerConfig(**cfg_dict)
 
        self.backbone = SpecFormer(cfg)
        self.backbone.load_state_dict(ckpt["model"], strict=False)
        print("[SpectrumBackbone] loaded")
 
        # ✅ 冻结
        for p in self.backbone.parameters():
            p.requires_grad = False
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) 或 (B, L, 1)
        返回: (B, seq_len, 768)  — 全部 token embedding
        """
        outputs = self.backbone(x)          # dict
        return outputs["embedding"]         # (B, seq_len, 768)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# AstroCLIP 完整模型（论文 Section 3.3）
# ══════════════════════════════════════════════════════════════════════════════
class AstroClipModelV3(nn.Module):
    """
    冻结 backbone，只训练两个 CrossAttentionHead。
    embedding 维度 = 512（论文），logit_scale 固定为 15.5。
    """
    IMG_TOKEN_DIM  = 1024   # vit_large token dim
    SPEC_TOKEN_DIM = 768    # SpecFormer token dim
    EMBED_DIM      = 512    # 论文输出维度
    LOGIT_SCALE    = 15.5   # 论文 B3.2 固定值
 
    def __init__(self, image_backbone: nn.Module, spectrum_backbone: nn.Module):
        super().__init__()
        self.image_encoder    = image_backbone
        self.spectrum_encoder = spectrum_backbone
 
        # 论文：4 heads，embed_dim=512，各模态独立的交叉注意力头
        self.img_head  = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.IMG_TOKEN_DIM)
        self.spec_head = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.SPEC_TOKEN_DIM)
 
        self.criterion = CLIPLoss()
 
    def forward(self, images: torch.Tensor, spectrum: torch.Tensor):
        # backbone 冻结，不需要梯度
        with torch.no_grad():
            img_tokens  = self.image_encoder(images)     # (B, N, 1024)
            spec_tokens = self.spectrum_encoder(spectrum) # (B, S, 768)
 
        img_feat  = self.img_head(img_tokens)             # (B, 512)
        spec_feat = self.spec_head(spec_tokens)           # (B, 512)
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
    dm = AstroClipDataloader()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
 
    # 2️⃣ 构建 backbone
    image_backbone    = ImageBackboneWrapper(IMAGE_CKPT)
    spectrum_backbone = SpectrumBackboneWrapper(SPECTRUM_CKPT)
 
    # 3️⃣ 构建完整模型
    model = AstroClipModelV3(image_backbone, spectrum_backbone)
    model.to(device)
 
    # 4️⃣ 可训练参数统计（只有两个 CrossAttentionHead）
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Total params:     {total_params:,}")
    print(f"✅ Trainable params: {trainable_params:,}\n")
 
    # 5️⃣ Optimizer（论文：AdamW, lr=1e-4, weight_decay=0.01）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
 
    num_epochs = 100
 
    # 6️⃣ Cosine scheduler（论文）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
 
    # 7️⃣ Early stopping
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