"""
umap_visualize.py
使用 trainer3 训练好的模型（AstroClipModelV3）对验证集生成 embedding，
将图像和光谱特征拼接后，用 UMAP 降至 2 维并可视化保存。

权重文件：outputs1234/trained_models/astroclip_official_best_new.pt
模型类：AstroClipModelV3（定义在 trainer3.py）
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── 路径 ─────────────────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)

MODEL_PATH  = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models/astroclip_trainer3_best.pt")
OUTPUT_PATH = os.path.join(ASTROCLIP_ROOT, "outputs1234/umap_visualization.png")

# ── 导入模型类（直接从 trainer3 复用）────────────────────────────────────────
sys.path.insert(0, os.path.join(ASTROCLIP_ROOT, "astroclip"))
from trainer3 import (
    AstroClipModelV3,
    ImageBackboneWrapper,
    SpectrumBackboneWrapper,
    IMAGE_CKPT,
    SPECTRUM_CKPT,
)
from astroclip.data.datamodule import AstroClipDataloader


# ── 1. 提取 embedding ─────────────────────────────────────────────────────────
def extract_embeddings(model, dataloader, device):
    """
    对 dataloader 里的每个 batch 提取图像和光谱 embedding，
    L2 归一化后拼接，返回 (N, 1024) 的融合特征矩阵。

    拼接方式：[img_feat (512) | spec_feat (512)] → (1024,)
    每个数据对只对应一个向量，保留了两个模态的信息。
    """
    model.eval()
    all_fused = []

    print("🔍 正在提取 embedding...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images  = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)

            img_feat, spec_feat = model(images, spectra)

            # L2 归一化
            img_feat  = F.normalize(img_feat,  dim=-1)  # (B, 512)
            spec_feat = F.normalize(spec_feat, dim=-1)  # (B, 512)

            # 拼接两个模态的特征
            fused = torch.cat([img_feat, spec_feat], dim=-1)  # (B, 1024)
            all_fused.append(fused.cpu())

    fused_embeddings = torch.cat(all_fused, dim=0).numpy()  # (N, 1024)
    print(f"✅ 提取完成，共 {fused_embeddings.shape[0]} 个样本，特征维度 {fused_embeddings.shape[1]}")
    return fused_embeddings


# ── 2. UMAP 降维 ──────────────────────────────────────────────────────────────
def run_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    将高维 embedding 降至 2 维。
    """
    try:
        import umap
    except ImportError:
        raise ImportError("请先安装 umap-learn：pip install umap-learn --break-system-packages")

    print("🗺️  正在运行 UMAP 降维（这可能需要几分钟）...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,    # 局部结构的邻居数，越大越关注全局结构
        min_dist=0.1,      # 点之间的最小距离，越小聚类越紧密
        metric="cosine",   # 和 CLIP 训练时一致，用余弦距离
        random_state=42,
        verbose=True,
    )
    embedding_2d = reducer.fit_transform(embeddings)  # (N, 2)
    print("✅ UMAP 降维完成")
    return embedding_2d


# ── 3. 可视化并保存 ───────────────────────────────────────────────────────────
def visualize(embedding_2d: np.ndarray, save_path: str):
    """
    将 2D embedding 可视化，按点密度着色，保存为 PNG。
    没有类别标签，用密度来体现潜在结构。
    """
    x, y = embedding_2d[:, 0], embedding_2d[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 用散点图，alpha 控制透明度以体现密度
    scatter = ax.scatter(
        x, y,
        s=6,            # 点的大小
        alpha=0.7,      # 透明度，重叠处颜色更深
        c=y,            # 用 y 坐标着色，让结构更清晰
        cmap="viridis",
        linewidths=0,
    )

    plt.colorbar(scatter, ax=ax, label="UMAP dim 2")
    ax.set_title("AstroCLIP UMAP Visualization\n(fused image + spectrum embeddings)", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 可视化图像已保存至：{save_path}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    # 1️⃣ 数据加载（使用验证集作为测试集）
    dm          = AstroClipDataloader()
    val_loader  = dm.val_dataloader()

    # 2️⃣ 构建模型
    print("\n📦 正在构建模型...")
    image_backbone    = ImageBackboneWrapper(IMAGE_CKPT)
    spectrum_backbone = SpectrumBackboneWrapper(SPECTRUM_CKPT)
    model = AstroClipModelV3(image_backbone, spectrum_backbone)

    # 3️⃣ 加载权重
    print(f"📂 加载权重: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到权重文件：{MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("✅ 权重加载成功")

    # 4️⃣ 提取 embedding（图像 + 光谱拼接）
    embeddings = extract_embeddings(model, val_loader, device)  # (N, 1024)

    # 5️⃣ UMAP 降维
    embedding_2d = run_umap(embeddings)  # (N, 2)

    # 6️⃣ 保存 2D embedding 数据
    npy_path = OUTPUT_PATH.replace(".png", "_2d.npy")
    np.save(npy_path, embedding_2d)
    print(f"✅ 2D embedding 已保存至：{npy_path}")

    # 7️⃣ 可视化并保存
    visualize(embedding_2d, OUTPUT_PATH)


if __name__ == "__main__":
    main()