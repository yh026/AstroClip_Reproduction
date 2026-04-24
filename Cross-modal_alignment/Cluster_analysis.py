"""
cluster_analysis.py
基于 umap_visualize.py 生成的 2D embedding 和原始数据，进行四步分析：
1. HDBSCAN 聚类并可视化
2. 每个类选代表点，在 UMAP 图上替换为实际图像
3. 每个类代表点的图像和光谱
4. 按红移值着色的 UMAP 图
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
import torch
import torch.nn.functional as F

# ── 路径 ─────────────────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)
sys.path.insert(0, os.path.join(ASTROCLIP_ROOT, "astroclip"))

EMBEDDING_2D_PATH = os.path.join(ASTROCLIP_ROOT, "outputs1234/umap_visualization_2d.npy")
OUTPUT_DIR        = os.path.join(ASTROCLIP_ROOT, "outputs1234/cluster_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from astroclip.data.datamodule import AstroClipDataloader
from trainer3 import (
    AstroClipModelV3,
    ImageBackboneWrapper,
    SpectrumBackboneWrapper,
    IMAGE_CKPT,
    SPECTRUM_CKPT,
)

MODEL_PATH = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models/astroclip_trainer3_best.pt")

# ── 调色板（最多支持 20 个簇）────────────────────────────────────────────────
CLUSTER_CMAP = plt.cm.get_cmap("tab20")


# ══════════════════════════════════════════════════════════════════════════════
# 数据收集：原始图像、光谱、红移
# ══════════════════════════════════════════════════════════════════════════════
def collect_raw_data(dataloader):
    """
    遍历 dataloader，收集：
    - raw_images:   list of (3, H, W) tensor，用于可视化
    - raw_spectra:  list of (L,) tensor
    - redshifts:    list of float（如果数据集有 redshift 列）
    返回三个 numpy 数组。
    """
    raw_images, raw_spectra, redshifts = [], [], []
    has_redshift = None

    print("📦 收集原始数据（图像 / 光谱 / 红移）...")
    for batch in tqdm(dataloader, desc="Collecting"):
        imgs = batch["image"]           # (B, C, H, W)  已经过 collator 处理
        specs = batch["spectrum"]       # (B, L)

        for i in range(imgs.size(0)):
            raw_images.append(imgs[i].cpu().numpy())    # (C, H, W)
            raw_spectra.append(specs[i].cpu().numpy())  # (L,)

        # 红移：只在第一个 batch 判断一次
        if has_redshift is None:
            has_redshift = "redshift" in batch
        if has_redshift:
            z = batch["redshift"]
            for i in range(z.size(0)):
                redshifts.append(float(z[i]))

    raw_images  = np.stack(raw_images)   # (N, C, H, W)
    raw_spectra = np.stack(raw_spectra)  # (N, L)
    redshifts   = np.array(redshifts) if has_redshift else None

    print(f"  图像: {raw_images.shape}, 光谱: {raw_spectra.shape}")
    if redshifts is not None:
        print(f"  红移: {redshifts.shape}, min={redshifts.min():.3f}, max={redshifts.max():.3f}")
    else:
        print("  ⚠️ 数据集中未找到 redshift 列，步骤4将跳过")
    return raw_images, raw_spectra, redshifts


# ══════════════════════════════════════════════════════════════════════════════
# 图像预处理：(C, H, W) float → (H, W, 3) uint8，适合 imshow
# ══════════════════════════════════════════════════════════════════════════════
def tensor_to_rgb(img_chw: np.ndarray) -> np.ndarray:
    """
    img_chw: (3, H, W) float，可能已经 Z-score 归一化
    返回:    (H, W, 3) uint8 [0, 255]
    """
    img = img_chw.transpose(1, 2, 0)           # (H, W, 3)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 1：HDBSCAN 聚类 + 可视化
# ══════════════════════════════════════════════════════════════════════════════
def step1_hdbscan(embedding_2d: np.ndarray) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        raise ImportError("请先安装 hdbscan：pip install hdbscan")

    print("\n🔵 步骤1：HDBSCAN 聚类...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,    # 调小：允许更小的簇存在
        min_samples=3,          # 调小：更容易成为核心点
        cluster_selection_epsilon=0.3,  # 合并距离阈值，调小让相近的簇不被合并
        cluster_selection_method="leaf",  # leaf模式比eom模式找到更多细粒度的簇
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"  发现 {n_clusters} 个簇，噪声点 {n_noise} 个")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(labels))

    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       s=4, c="lightgray", alpha=0.4, label="noise")
        else:
            color = CLUSTER_CMAP(lbl % 20)
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       s=8, color=color, alpha=0.7, label=f"Cluster {lbl}")

    ax.legend(loc="upper right", fontsize=24, markerscale=2,
              ncol=max(1, n_clusters // 10))
    ax.set_title(f"HDBSCAN Clustering ({n_clusters} clusters, {n_noise} noise)", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    save_path = os.path.join(OUTPUT_DIR, "step1_hdbscan_clusters.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 保存至：{save_path}")

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 2：每个簇选代表点，在 UMAP 图上显示实际图像
# ══════════════════════════════════════════════════════════════════════════════
def step2_image_embedding(embedding_2d: np.ndarray, labels: np.ndarray,
                           raw_images: np.ndarray, n_rep: int = 3):
    """
    每个簇选 n_rep 个代表点（离簇中心最近的点），
    在 UMAP 图上用实际图像缩略图替代散点。
    """
    print("\n🖼️  步骤2：代表点图像展示...")
    unique_labels = [l for l in sorted(set(labels)) if l != -1]

    fig, ax = plt.subplots(figsize=(20, 18))

    # 先画所有背景散点（灰色）
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
               s=3, c="lightgray", alpha=0.3, zorder=1)

    # 噪声点单独标出
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(embedding_2d[noise_mask, 0], embedding_2d[noise_mask, 1],
                   s=3, c="silver", alpha=0.2, zorder=2)

    for lbl in unique_labels:
        mask    = np.where(labels == lbl)[0]
        center  = embedding_2d[mask].mean(axis=0)
        color   = CLUSTER_CMAP(lbl % 20)

        # 画簇的散点底色
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=6, color=color, alpha=0.4, zorder=3)

        # 标注簇编号
        ax.text(center[0], center[1], str(lbl),
                fontsize=9, fontweight="bold", color=color,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

        # 选离中心最近的 n_rep 个点作为代表
        dists = np.linalg.norm(embedding_2d[mask] - center, axis=1)
        rep_indices = mask[np.argsort(dists)[:n_rep]]

        for idx in rep_indices:
            rgb = tensor_to_rgb(raw_images[idx])          # (H, W, 3)
            # 缩放图像到 32×32 缩略图
            from PIL import Image as PILImage
            thumb = np.array(PILImage.fromarray(rgb).resize((32, 32)))
            im    = OffsetImage(thumb, zoom=1.5)
            ab    = AnnotationBbox(im, embedding_2d[idx],
                                   frameon=True,
                                   bboxprops=dict(edgecolor=color, linewidth=1.5),
                                   zorder=6)
            ax.add_artist(ab)

    ax.set_title("UMAP with Representative Galaxy Images per Cluster", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    save_path = os.path.join(OUTPUT_DIR, "step2_image_embedding.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 保存至：{save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 3：每个簇代表点的图像和光谱
# ══════════════════════════════════════════════════════════════════════════════
def step3_cluster_gallery(embedding_2d: np.ndarray, labels: np.ndarray,
                           raw_images: np.ndarray, raw_spectra: np.ndarray,
                           n_rep: int = 5):
    """
    每个簇展示 n_rep 个代表点，每个代表点画：左图像 + 右光谱。
    每个簇保存一张图。
    """
    print("\n📊 步骤3：每个簇代表点的图像和光谱...")
    unique_labels = [l for l in sorted(set(labels)) if l != -1]

    for lbl in unique_labels:
        mask   = np.where(labels == lbl)[0]
        center = embedding_2d[mask].mean(axis=0)
        dists  = np.linalg.norm(embedding_2d[mask] - center, axis=1)
        rep_indices = mask[np.argsort(dists)[:n_rep]]

        fig, axes = plt.subplots(n_rep, 2,
                                 figsize=(10, 3 * n_rep),
                                 gridspec_kw={"width_ratios": [1, 2]})
        if n_rep == 1:
            axes = axes[np.newaxis, :]

        color = CLUSTER_CMAP(lbl % 20)
        fig.suptitle(f"Cluster {lbl}  ({len(mask)} galaxies)", fontsize=14,
                     color=color, fontweight="bold")

        for row, idx in enumerate(rep_indices):
            # 左：图像
            rgb = tensor_to_rgb(raw_images[idx])
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"Sample #{idx}", fontsize=8)
            axes[row, 0].axis("off")

            # 右：光谱
            spec = raw_spectra[idx]
            axes[row, 1].plot(spec, lw=0.8, color=color, alpha=0.85)
            axes[row, 1].set_ylabel("Flux", fontsize=8)
            axes[row, 1].set_xlabel("Wavelength bin", fontsize=8)
            axes[row, 1].tick_params(labelsize=7)
            axes[row, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"step3_cluster{lbl:02d}_gallery.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  ✅ 每个簇的 gallery 已保存至：{OUTPUT_DIR}/step3_cluster*_gallery.png")


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 4：按红移值着色的 UMAP 图
# ══════════════════════════════════════════════════════════════════════════════
def step4_redshift_coloring(embedding_2d: np.ndarray, redshifts: np.ndarray):
    if redshifts is None:
        print("\n⚠️  步骤4跳过：数据集中没有 redshift 列")
        return

    print("\n🌈 步骤4：按红移着色...")

    # 裁剪极端值，让颜色分布更均匀
    vmin = np.percentile(redshifts, 2)
    vmax = np.percentile(redshifts, 98)

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=redshifts,
        cmap="plasma",
        vmin=vmin, vmax=vmax,
        s=5, alpha=0.6,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Redshift z", fontsize=12)

    ax.set_title("UMAP colored by Redshift", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    save_path = os.path.join(OUTPUT_DIR, "step4_redshift_coloring.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 保存至：{save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── 加载 2D embedding ──
    if not os.path.exists(EMBEDDING_2D_PATH):
        raise FileNotFoundError(
            f"找不到 2D embedding 文件：{EMBEDDING_2D_PATH}\n"
            "请先运行 umap_visualize.py 生成该文件。"
        )
    embedding_2d = np.load(EMBEDDING_2D_PATH)   # (N, 2)
    print(f"✅ 加载 2D embedding：{embedding_2d.shape}")

    # ── 加载原始数据（图像 / 光谱 / 红移）──
    # 需要把 redshift 也加入 columns
    dm = AstroClipDataloader(columns=["image", "spectrum", "redshift"])
    val_loader = dm.val_dataloader()

    raw_images, raw_spectra, redshifts = collect_raw_data(val_loader)

    # 对齐：drop_last=True 可能让 dataloader 少一些样本
    N = min(len(embedding_2d), len(raw_images))
    embedding_2d = embedding_2d[:N]
    raw_images   = raw_images[:N]
    raw_spectra  = raw_spectra[:N]
    if redshifts is not None:
        redshifts = redshifts[:N]

    print(f"\n最终对齐样本数：{N}")

    # ── 四步分析 ──
    labels = step1_hdbscan(embedding_2d)
    step2_image_embedding(embedding_2d, labels, raw_images, n_rep=3)
    step3_cluster_gallery(embedding_2d, labels, raw_images, raw_spectra, n_rep=5)
    step4_redshift_coloring(embedding_2d, redshifts)

    print(f"\n🎉 全部完成！结果保存在：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()