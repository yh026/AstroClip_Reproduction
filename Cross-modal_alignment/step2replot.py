"""
step2_replot.py
重新生成 step2_image_embedding2.png：
- 散点 s=6，透明度降低
- 图片边长放大到原来的 1.5 倍（zoom=2.25）
- 每个簇按簇大小比例抽取代表点，随机分散选取（不集中在中心）
- 图片位置对应其数据点位置
- HDBSCAN 参数与 cluster_analysis.py 保持一致
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
from PIL import Image as PILImage
from matplotlib.lines import Line2D

# ── 路径 ─────────────────────────────────────────────────────────────────────
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
sys.path.insert(0, ASTROCLIP_ROOT)
sys.path.insert(0, os.path.join(ASTROCLIP_ROOT, "astroclip"))

EMBEDDING_2D_PATH = os.path.join(ASTROCLIP_ROOT, "outputs1234/umap_visualization_2d.npy")
OUTPUT_DIR        = os.path.join(ASTROCLIP_ROOT, "outputs1234/cluster_results")
SAVE_PATH         = os.path.join(OUTPUT_DIR, "step2_image_embedding2.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from astroclip.data.datamodule import AstroClipDataloader

CLUSTER_CMAP = plt.cm.get_cmap("tab20")


# ── 图像预处理 ────────────────────────────────────────────────────────────────
def tensor_to_rgb(img_chw: np.ndarray) -> np.ndarray:
    img = img_chw.transpose(1, 2, 0)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).clip(0, 255).astype(np.uint8)


# ── 收集原始图像 ──────────────────────────────────────────────────────────────
def collect_images(dataloader):
    raw_images = []
    print("📦 收集图像数据...")
    for batch in tqdm(dataloader, desc="Collecting"):
        imgs = batch["image"]
        for i in range(imgs.size(0)):
            raw_images.append(imgs[i].cpu().numpy())
    return np.stack(raw_images)


# ── HDBSCAN（与 cluster_analysis.py 完全一致）────────────────────────────────
def run_hdbscan(embedding_2d: np.ndarray) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        raise ImportError("请先安装 hdbscan：pip install hdbscan")

    print("🔵 运行 HDBSCAN（参数与 cluster_analysis.py 一致）...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        cluster_selection_epsilon=0.3,
        cluster_selection_method="leaf",
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"  发现 {n_clusters} 个簇，噪声点 {n_noise} 个")
    return labels


# ── 按比例随机抽取代表点 ──────────────────────────────────────────────────────
def sample_representatives(labels: np.ndarray,
                            base_count: int = 5,
                            max_count: int = 20,
                            rng: np.random.Generator = None) -> dict:
    """
    每个簇按簇大小比例抽取代表点索引，随机分散选取。

    base_count: 最小簇对应的最少图片数
    max_count:  最大簇对应的最多图片数
    返回: {label: [idx, ...]}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unique_labels = [l for l in sorted(set(labels)) if l != -1]
    sizes = {l: int(np.sum(labels == l)) for l in unique_labels}
    min_size = min(sizes.values())
    max_size = max(sizes.values())

    rep_dict = {}
    for lbl in unique_labels:
        mask = np.where(labels == lbl)[0]
        size = sizes[lbl]

        # 线性插值：小簇取 base_count，大簇取 max_count
        if max_size == min_size:
            n_rep = base_count
        else:
            ratio = (size - min_size) / (max_size - min_size)
            n_rep = int(base_count + ratio * (max_count - base_count))
        n_rep = min(n_rep, size)  # 不超过簇大小

        # 随机打散后取前 n_rep 个，保证分散
        chosen = rng.choice(mask, size=n_rep, replace=False)
        rep_dict[lbl] = chosen

    return rep_dict


# ── 主绘图函数 ────────────────────────────────────────────────────────────────
def plot_step2(embedding_2d: np.ndarray, labels: np.ndarray,
               raw_images: np.ndarray, save_path: str):

    unique_labels = [l for l in sorted(set(labels)) if l != -1]
    rep_dict = sample_representatives(labels, base_count=5, max_count=20)

    # ✅ 放大画布（给右侧 legend 留空间）
    fig, ax = plt.subplots(figsize=(24, 21))

    # ── 背景散点 ──
    noise_mask = labels == -1
    ax.scatter(embedding_2d[noise_mask, 0], embedding_2d[noise_mask, 1],
               s=36, c="lightgray", alpha=0.5, zorder=1)

    for lbl in unique_labels:
        mask  = labels == lbl
        color = CLUSTER_CMAP(lbl % 20)
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=36, color=color, alpha=0.7, zorder=2)

    # ── 代表点图像 ──
    for lbl in unique_labels:
        color     = CLUSTER_CMAP(lbl % 20)
        rep_idxs  = rep_dict[lbl]

        for idx in rep_idxs:
            rgb   = tensor_to_rgb(raw_images[idx])
            thumb = np.array(PILImage.fromarray(rgb).resize((32, 32)))

            im = OffsetImage(thumb, zoom=2.25)
            ab = AnnotationBbox(
                im,
                embedding_2d[idx],
                frameon=True,
                bboxprops=dict(edgecolor=color, linewidth=1.5),
                pad=0.1,
                zorder=6,
            )
            ax.add_artist(ab)

    # ── 标题 & 坐标 ──
    ax.set_title(
        "UMAP with Representative Galaxy Images per Cluster\n"
        "(sampled proportionally, placed at data point location)",
        fontsize=36
    )
    ax.set_xlabel("UMAP dim 1", fontsize=32)
    ax.set_ylabel("UMAP dim 2", fontsize=32)
    ax.tick_params(axis='both', labelsize=28)

    # ── 构造 legend（图外）──
    legend_elements = []

    # noise
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w',
               label='noise',
               markerfacecolor='lightgray',
               markersize=15, alpha=0.6)
    )

    # clusters
    for lbl in unique_labels:
        color = CLUSTER_CMAP(lbl % 20)
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   label=f'Cluster {lbl}',
                   markerfacecolor=color,
                   markersize=15)
        )

    # 👉 根据 cluster 数量自动分列（很关键）
    ncol = 1
    if len(unique_labels) > 20:
        ncol = 2
    if len(unique_labels) > 40:
        ncol = 3

    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),   # ✅ 放到右侧外部
        fontsize=28,
        title="Clusters",
        title_fontsize=30,
        ncol=ncol,
        borderaxespad=0.
    )

    # ── 布局 & 保存 ──
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # ✅ 防裁剪
    plt.close()

    print(f"✅ 保存至：{save_path}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    # 加载 2D embedding
    if not os.path.exists(EMBEDDING_2D_PATH):
        raise FileNotFoundError(
            f"找不到 2D embedding：{EMBEDDING_2D_PATH}\n"
            "请先运行 umap_visualize.py"
        )
    embedding_2d = np.load(EMBEDDING_2D_PATH)
    print(f"✅ 加载 2D embedding：{embedding_2d.shape}")

    # 加载图像
    dm         = AstroClipDataloader()
    val_loader = dm.val_dataloader()
    raw_images = collect_images(val_loader)

    # 对齐样本数
    N            = min(len(embedding_2d), len(raw_images))
    embedding_2d = embedding_2d[:N]
    raw_images   = raw_images[:N]
    print(f"对齐样本数：{N}")

    # HDBSCAN
    labels = run_hdbscan(embedding_2d)

    # 绘图
    print("🖼️  生成 step2_image_embedding2.png ...")
    plot_step2(embedding_2d, labels, raw_images, SAVE_PATH)
    print("🎉 完成！")


if __name__ == "__main__":
    main()