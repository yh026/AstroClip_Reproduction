"""
cross3.py
跨模态对齐评估脚本：Image-to-Spectrum & Spectrum-to-Image Retrieval
针对 trainer_official.py（官方backbone + CrossAttentionHead，论文结构）
权重文件：outputs1234/trained_models/astroclip_official_best_new.pt
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── 1. 路径与依赖导入 ────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)

MODEL_PATH = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models/astroclip_official_best_new.pt")

from astroclip.data.datamodule import AstroClipDataloader
from astroclip.models.astroclip import ImageHead, SpectrumHead

# 复用 trainer_official.py 中定义的所有类
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from trainer import (
        CrossAttentionHead,
        OfficialImageEncoder,
        OfficialSpectrumEncoder,
        AstroClipOfficial,
    )
except ImportError:
    raise ImportError("❌ 找不到 trainer_official.py，请确保该脚本与 trainer_official.py 放在同一目录下！")


# ── 2. 评估核心逻辑 ────────────────────────────────────────────────────────
def evaluate_cross_modal_retrieval(model, dataloader, device):
    """
    计算跨模态检索的 Recall@1, Recall@5, Recall@10
    """
    model.eval()

    all_image_embeds = []
    all_spec_embeds  = []

    print("\n🔍 第一阶段: 正在提取全局跨模态特征 (Extracting embeddings)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Feature Extraction"):
            images  = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)

            # AstroClipOfficial.forward() 返回 (img_feat, spec_feat)
            img_features, spec_features = model(images, spectra)

            # L2 归一化
            img_features  = F.normalize(img_features,  dim=-1)
            spec_features = F.normalize(spec_features, dim=-1)

            all_image_embeds.append(img_features.cpu())
            all_spec_embeds.append(spec_features.cpu())

    Z_im = torch.cat(all_image_embeds, dim=0)
    Z_sp = torch.cat(all_spec_embeds,  dim=0)

    N = Z_im.shape[0]
    print(f"✅ 提取完成！测试集样本总数 (Total evaluate samples): {N}")

    # ── 计算相似度矩阵 ──
    print("\n🧮 第二阶段: 计算全局 N x N 相似度矩阵...")

    calc_device = device if N <= 25000 else torch.device("cpu")
    if calc_device.type == "cpu" and device.type == "cuda":
        print("⚠️ 样本量较大，相似度矩阵计算将回退至 CPU。")

    Z_im = Z_im.to(calc_device)
    Z_sp = Z_sp.to(calc_device)

    similarity_matrix = Z_im @ Z_sp.T   # (N, N)

    K_vals     = [1, 5, 10]
    i2s_recall = {k: 0.0 for k in K_vals}
    s2i_recall = {k: 0.0 for k in K_vals}

    targets = torch.arange(N, device=calc_device).unsqueeze(1)

    # ── 执行检索与统计 ──
    print("📊 第三阶段: 统计 Top-K 命中率 (Recall@K)...")

    _, i2s_topk_indices = similarity_matrix.topk(max(K_vals), dim=1, largest=True, sorted=True)
    _, s2i_topk_indices = similarity_matrix.t().topk(max(K_vals), dim=1, largest=True, sorted=True)

    for k in K_vals:
        i2s_correct = (i2s_topk_indices[:, :k] == targets).any(dim=1).float().sum().item()
        s2i_correct = (s2i_topk_indices[:, :k] == targets).any(dim=1).float().sum().item()

        i2s_recall[k] = (i2s_correct / N) * 100
        s2i_recall[k] = (s2i_correct / N) * 100

    # ── 打印最终报告 ──
    print("\n" + "═"*50)
    print("🏆 跨模态检索最终评估报告 (Cross-Modal Retrieval)")
    print("═"*50)

    print("🌌 Image-to-Spectrum (I2S) Retrieval:")
    for k in K_vals:
        print(f"   ➤ Recall@{k:<2d}: {i2s_recall[k]:>6.2f}%")

    print("\n📸 Spectrum-to-Image (S2I) Retrieval:")
    for k in K_vals:
        print(f"   ➤ Recall@{k:<2d}: {s2i_recall[k]:>6.2f}%")
    print("═"*50 + "\n")

    return i2s_recall, s2i_recall


# ── 3. 主程序入口 ────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动评估脚本 | 当前运行设备 (Device): {device}")

    # 数据加载（使用验证集）
    dm = AstroClipDataloader()
    test_loader = dm.val_dataloader()

    # 构建模型（和 trainer_official.py 完全一致）
    print("\n📦 正在构建网络结构...")
    model = AstroClipOfficial()

    # 加载训练好的权重
    print(f"📂 正在加载训练好的权重: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到权重文件 {MODEL_PATH}，请确认 trainer_official.py 是否已经训练完毕！"
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("✅ 权重加载成功！")

    # 开始评估
    evaluate_cross_modal_retrieval(model, test_loader, device)


if __name__ == "__main__":
    main()