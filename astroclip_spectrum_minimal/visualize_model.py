"""
SpecFormer Visualization Suite
==============================
Load best.pt and the dataset, generate a full set of figures to explain
the trained model to collaborators / reviewers.

Usage:
    python visualize_model.py \
        --checkpoint /path/to/outputs_specformer/best.pt \
        --data_path  /path/to/hf_dataset \
        --out_dir    ./figures

Generates:
    1. loss_curves.png           – train / val loss over epochs
    2. reconstruction_gallery.png – masked → reconstructed vs original (multiple samples)
    3. embedding_tsne.png        – t-SNE of final-layer embeddings
    4. embedding_pca.png         – PCA of final-layer embeddings
    5. layer_progression.png     – how embeddings evolve across transformer layers
    6. attention_heatmap.png     – attention weights of each head (layer 0 & last layer)
    7. parameter_health.png      – weight distribution per layer (histogram + stats)
    8. mask_illustration.png     – illustrate the masking strategy on one sample
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from matplotlib import cm

# ── make imports work when this file sits next to model.py ──────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SpecFormer, SpecFormerConfig
from data import DataConfig, SpectrumDataModule

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ============================================================================
#  Helpers
# ============================================================================

def load_model_and_data(ckpt_path, data_path, device="cpu"):
    """Load best.pt checkpoint and return model + val loader."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    history = ckpt.get("history", {})

    model_cfg = SpecFormerConfig(**cfg["model"])
    model = SpecFormer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_cfg = DataConfig(**{**cfg["data"], "path": data_path})
    dm = SpectrumDataModule(data_cfg)
    dm.setup()

    return model, dm, cfg, history


def flatten_for_plot(x: torch.Tensor) -> torch.Tensor:
    """Remove prepended summary token and flatten sliced tokens back to 1-D."""
    return x[:, 1:, 2:].reshape(x.size(0), -1)


def get_samples(loader, n=16, device="cpu"):
    """Grab n samples from a loader."""
    spectra = []
    for batch in loader:
        spectra.append(batch["spectrum"])
        if sum(s.size(0) for s in spectra) >= n:
            break
    return torch.cat(spectra, dim=0)[:n].to(device)


@torch.no_grad()
def extract_attention_weights(model, x_preprocessed):
    """
    Manual forward pass that captures attention weights from each layer.
    Returns list of (B, num_heads, T, T) tensors.
    """
    bsz, seqlen, _ = x_preprocessed.shape
    pos = torch.arange(0, seqlen, dtype=torch.long, device=x_preprocessed.device)
    h = model.dropout(model.data_embed(x_preprocessed) + model.position_embed(pos))

    attn_weights_all = []
    for block in model.blocks:
        # ── self-attention with explicit weight extraction ──
        residual = h
        x_ln = block.ln1(h)
        bsz_, sl_, dim_ = x_ln.shape
        head_dim = dim_ // block.attn.num_heads

        q, k, v = block.attn.qkv(x_ln).split(dim_, dim=2)
        q = q.view(bsz_, sl_, block.attn.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz_, sl_, block.attn.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz_, sl_, block.attn.num_heads, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = F.softmax(att, dim=-1)
        attn_weights_all.append(att.detach().cpu())

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz_, sl_, dim_)
        y = block.attn.resid_drop(block.attn.proj(y))
        h = residual + y

        # ── MLP ──
        h = h + block.mlp(block.ln2(h))

    return attn_weights_all


# ============================================================================
#  Figure 1: Loss Curves
# ============================================================================

def plot_loss_curves(history, out_dir):
    """Train/val loss curve from history dict."""
    train_loss = history.get("train_loss_epoch", [])
    val_loss = history.get("val_loss_epoch", [])
    if not train_loss:
        print("[SKIP] No training history found.")
        return

    epochs = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss curves
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#2196F3", markersize=5)
    ax1.plot(epochs[:len(val_loss)], val_loss, "s-", label="Val Loss", color="#FF5722", markersize=5)
    if history.get("test_loss"):
        ax1.axhline(history["test_loss"][-1], ls="--", color="#4CAF50", label=f"Test Loss = {history['test_loss'][-1]:.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (masked region)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: learning rate
    lr_history = history.get("lr_epoch_end", [])
    if lr_history:
        ax2.plot(epochs[:len(lr_history)], lr_history, "o-", color="#9C27B0", markersize=5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No LR history", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("Figure 1: Training Dynamics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "1_loss_curves.png")
    plt.close(fig)
    print("[OK] 1_loss_curves.png")


# ============================================================================
#  Figure 2: Masking Strategy Illustration
# ============================================================================

@torch.no_grad()
def plot_mask_illustration(model, spectra, out_dir):
    """Show one sample: original → masked → what the model sees."""
    spec = spectra[:1]
    target = model.preprocess(spec)
    masked = model.mask_sequence(target)

    orig_flat = flatten_for_plot(target)[0].cpu().numpy()
    mask_flat = flatten_for_plot(masked)[0].cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(orig_flat, color="#2196F3", linewidth=0.6)
    axes[0].set_title("(a) Original preprocessed spectrum (after slicing)")
    axes[0].set_ylabel("Normalised flux")

    # Highlight masked regions
    mask_indicator = (mask_flat == 0) & (orig_flat != 0)
    axes[1].plot(mask_flat, color="#FF9800", linewidth=0.6)
    axes[1].fill_between(range(len(mask_flat)), orig_flat.min(), orig_flat.max(),
                         where=mask_indicator, alpha=0.25, color="red", label="Masked region")
    axes[1].set_title("(b) Masked input (6 random chunks zeroed out)")
    axes[1].set_ylabel("Normalised flux")
    axes[1].legend(loc="upper right")

    # Difference
    diff = np.abs(orig_flat - mask_flat)
    axes[2].fill_between(range(len(diff)), 0, diff, color="#F44336", alpha=0.6)
    axes[2].set_title("(c) Masked positions (model must reconstruct these)")
    axes[2].set_xlabel("Flattened token index")
    axes[2].set_ylabel("|Original − Masked|")

    fig.suptitle("Figure 2: Masking Strategy", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "2_mask_illustration.png")
    plt.close(fig)
    print("[OK] 2_mask_illustration.png")


# ============================================================================
#  Figure 3: Reconstruction Gallery
# ============================================================================

@torch.no_grad()
def plot_reconstruction_gallery(model, spectra, out_dir, n_show=8, win=10):
    """Show original / masked / reconstructed for multiple samples."""
    n_show = min(n_show, spectra.size(0))
    target = model.preprocess(spectra[:n_show])
    masked = model.mask_sequence(target)
    outputs = model.forward_preprocessed(masked)
    recon = outputs["reconstructions"]

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        t = flatten_for_plot(target[i:i+1])[0].cpu().numpy()
        m = flatten_for_plot(masked[i:i+1])[0].cpu().numpy()
        r = flatten_for_plot(recon[i:i+1])[0].cpu().numpy()

        # Moving average for cleaner visualization
        def smooth(arr, w=win):
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w)/w, mode="valid")

        xs_t = range(len(smooth(t)))
        axes[i].plot(xs_t, smooth(t), label="Original", color="#2196F3", linewidth=1.2, alpha=0.8)
        axes[i].plot(xs_t, smooth(m), label="Masked input", color="#FF9800", linewidth=1.0, alpha=0.6)
        axes[i].plot(xs_t, smooth(r), label="Reconstructed", color="#4CAF50", linewidth=1.2, alpha=0.8, linestyle="--")
        axes[i].set_ylabel(f"Sample {i}")
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Flattened token index (smoothed)")
    fig.suptitle("Figure 3: Reconstruction Gallery — Original vs Masked vs Reconstructed",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "3_reconstruction_gallery.png")
    plt.close(fig)
    print("[OK] 3_reconstruction_gallery.png")


# ============================================================================
#  Figure 4: Reconstruction Error Heatmap
# ============================================================================

@torch.no_grad()
def plot_reconstruction_error(model, spectra, out_dir, n_show=16):
    """2-D heatmap: sample × position, colour = |recon − target|."""
    n_show = min(n_show, spectra.size(0))
    target = model.preprocess(spectra[:n_show])
    masked = model.mask_sequence(target)
    outputs = model.forward_preprocessed(masked)
    recon = outputs["reconstructions"]

    t_flat = flatten_for_plot(target).cpu().numpy()
    r_flat = flatten_for_plot(recon).cpu().numpy()
    m_flat = flatten_for_plot(masked).cpu().numpy()

    error = np.abs(t_flat - r_flat)
    mask_indicator = (m_flat == 0).astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    im1 = ax1.imshow(mask_indicator, aspect="auto", cmap="Reds", interpolation="nearest")
    ax1.set_title("(a) Mask locations (red = masked)")
    ax1.set_ylabel("Sample")
    plt.colorbar(im1, ax=ax1, shrink=0.6)

    im2 = ax2.imshow(error, aspect="auto", cmap="hot", interpolation="nearest")
    ax2.set_title("(b) Reconstruction error |target − reconstructed|")
    ax2.set_ylabel("Sample")
    ax2.set_xlabel("Flattened token index")
    plt.colorbar(im2, ax=ax2, shrink=0.6)

    fig.suptitle("Figure 4: Reconstruction Error Heatmap", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "4_reconstruction_error.png")
    plt.close(fig)
    print("[OK] 4_reconstruction_error.png")


# ============================================================================
#  Figure 5: t-SNE & PCA of Embeddings
# ============================================================================

@torch.no_grad()
def plot_embedding_space(model, loader, out_dir, device="cpu", max_samples=500):
    """Collect final embeddings and show t-SNE + PCA (2-D) coloured by mean flux."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    embeddings = []
    mean_flux = []

    for batch in loader:
        spec = batch["spectrum"].to(device)
        preprocessed = model.preprocess(spec)
        outputs = model.encode(preprocessed, preprocessed=True)
        # Mean-pool over token dimension
        emb = outputs["embedding"].mean(dim=1).cpu()
        embeddings.append(emb)
        mean_flux.append(spec.mean(dim=-1).cpu() if spec.ndim == 2 else spec.mean(dim=(-2, -1)).cpu())
        if sum(e.size(0) for e in embeddings) >= max_samples:
            break

    embeddings = torch.cat(embeddings, dim=0)[:max_samples].numpy()
    mean_flux = torch.cat(mean_flux, dim=0)[:max_samples].numpy()

    # Normalise colour
    colors = (mean_flux - mean_flux.min()) / (mean_flux.max() - mean_flux.min() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(embeddings)
    sc1 = ax1.scatter(pca_proj[:, 0], pca_proj[:, 1], c=colors, cmap="viridis", s=12, alpha=0.7)
    ax1.set_title(f"PCA  (var explained: {pca.explained_variance_ratio_.sum():.1%})")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    plt.colorbar(sc1, ax=ax1, label="Mean flux (normalised)")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, max_samples // 4), random_state=42, max_iter=1000)
    tsne_proj = tsne.fit_transform(embeddings)
    sc2 = ax2.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, cmap="viridis", s=12, alpha=0.7)
    ax2.set_title("t-SNE")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    plt.colorbar(sc2, ax=ax2, label="Mean flux (normalised)")

    fig.suptitle("Figure 5: Learned Embedding Space (mean-pooled, coloured by mean flux)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "5_embedding_space.png")
    plt.close(fig)
    print("[OK] 5_embedding_space.png")


# ============================================================================
#  Figure 6: Layer-wise Embedding Progression
# ============================================================================

@torch.no_grad()
def plot_layer_progression(model, spectra, out_dir, device="cpu"):
    """Show how token embeddings evolve across layers using PCA."""
    from sklearn.decomposition import PCA

    preprocessed = model.preprocess(spectra[:8])
    outputs = model.encode(preprocessed, preprocessed=True, return_all_layers=True)

    layers_data = []
    labels = []

    # Token embedding (before blocks)
    te = outputs["token_embedding"].mean(dim=0).cpu().numpy()  # (T, D)
    layers_data.append(te)
    labels.append("Token Embed")

    # Each block output
    for idx, layer_emb in enumerate(outputs["all_layer_embeddings"]):
        le = layer_emb.mean(dim=0).cpu().numpy()
        layers_data.append(le)
        labels.append(f"Block {idx+1}")

    # Final embedding
    fe = outputs["embedding"].mean(dim=0).cpu().numpy()
    layers_data.append(fe)
    labels.append("Final LN")

    n = len(layers_data)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    # Fit PCA on final embedding for consistent axes
    pca = PCA(n_components=2)
    pca.fit(layers_data[-1])

    for i, (data, label) in enumerate(zip(layers_data, labels)):
        proj = pca.transform(data)
        token_ids = np.arange(data.shape[0])
        axes[i].scatter(proj[:, 0], proj[:, 1], c=token_ids, cmap="coolwarm", s=8, alpha=0.7)
        axes[i].set_title(label, fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide extra axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 6: Embedding Progression Across Layers\n(PCA projection, colour = token position)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "6_layer_progression.png")
    plt.close(fig)
    print("[OK] 6_layer_progression.png")


# ============================================================================
#  Figure 7: Attention Heatmaps
# ============================================================================

@torch.no_grad()
def plot_attention_heatmaps(model, spectra, out_dir):
    """Show attention maps from first and last transformer layer."""
    preprocessed = model.preprocess(spectra[:1])
    attn_list = extract_attention_weights(model, preprocessed)

    layers_to_show = [0, len(attn_list) - 1]
    num_heads = attn_list[0].shape[1]
    n_heads_show = min(num_heads, 6)

    fig, axes = plt.subplots(len(layers_to_show), n_heads_show,
                             figsize=(3.5 * n_heads_show, 3.5 * len(layers_to_show)))
    if len(layers_to_show) == 1:
        axes = axes[np.newaxis, :]

    # Subsample tokens for readability
    T = attn_list[0].shape[-1]
    step = max(1, T // 64)
    indices = list(range(0, T, step))

    for row, layer_idx in enumerate(layers_to_show):
        attn = attn_list[layer_idx][0]  # (num_heads, T, T)
        for col in range(n_heads_show):
            a = attn[col].numpy()
            a_sub = a[np.ix_(indices, indices)]
            axes[row, col].imshow(a_sub, cmap="Blues", aspect="auto", interpolation="nearest")
            axes[row, col].set_title(f"Layer {layer_idx+1}, Head {col+1}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Query token")
            if row == len(layers_to_show) - 1:
                axes[row, col].set_xlabel("Key token")

    fig.suptitle("Figure 7: Attention Weights (first & last layer, subsampled tokens)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "7_attention_heatmaps.png")
    plt.close(fig)
    print("[OK] 7_attention_heatmaps.png")


# ============================================================================
#  Figure 8: Parameter Health Check
# ============================================================================

def plot_parameter_health(model, out_dir):
    """Histogram of weight values per major component + statistics."""
    groups = {
        "data_embed": [],
        "position_embed": [],
        "blocks (attn)": [],
        "blocks (mlp)": [],
        "final_ln": [],
        "head": [],
    }

    for name, param in model.named_parameters():
        p = param.detach().cpu().flatten().numpy()
        if "data_embed" in name:
            groups["data_embed"].append(p)
        elif "position_embed" in name:
            groups["position_embed"].append(p)
        elif "blocks" in name and ("attn" in name or "ln1" in name):
            groups["blocks (attn)"].append(p)
        elif "blocks" in name and ("mlp" in name or "ln2" in name):
            groups["blocks (mlp)"].append(p)
        elif "final_ln" in name:
            groups["final_ln"].append(p)
        elif "head" in name:
            groups["head"].append(p)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]

    for i, (group_name, params) in enumerate(groups.items()):
        if not params:
            axes[i].text(0.5, 0.5, "No params", ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(group_name)
            continue
        all_p = np.concatenate(params)
        axes[i].hist(all_p, bins=100, color=colors[i], alpha=0.75, density=True)
        axes[i].set_title(f"{group_name}\n"
                          f"mean={all_p.mean():.4f}, std={all_p.std():.4f}\n"
                          f"min={all_p.min():.4f}, max={all_p.max():.4f}",
                          fontsize=10)
        axes[i].set_xlabel("Weight value")
        axes[i].set_ylabel("Density")
        axes[i].axvline(0, color="black", linewidth=0.5, alpha=0.5)

    fig.suptitle("Figure 8: Parameter Distribution Health Check",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "8_parameter_health.png")
    plt.close(fig)
    print("[OK] 8_parameter_health.png")


# ============================================================================
#  Figure 9: Layer-wise L2 Norm
# ============================================================================

@torch.no_grad()
def plot_layer_norms(model, spectra, out_dir):
    """Bar chart of mean token L2-norm at each layer."""
    preprocessed = model.preprocess(spectra[:16])
    outputs = model.encode(preprocessed, preprocessed=True, return_all_layers=True)

    names = ["Token Embed"]
    norms = [outputs["token_embedding"].norm(dim=-1).mean().item()]

    for idx, emb in enumerate(outputs["all_layer_embeddings"]):
        names.append(f"Block {idx+1}")
        norms.append(emb.norm(dim=-1).mean().item())

    names.append("Final (after LN)")
    norms.append(outputs["embedding"].norm(dim=-1).mean().item())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, norms, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(names))), edgecolor="white")
    ax.set_ylabel("Mean L2 Norm (per token)")
    ax.set_title("Figure 9: Layer-wise Embedding L2 Norm", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "9_layer_norms.png")
    plt.close(fig)
    print("[OK] 9_layer_norms.png")


# ============================================================================
#  Figure 10: Model Architecture Summary
# ============================================================================

def plot_model_summary(model, cfg, out_dir):
    """Text-based architecture summary figure."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_cfg = cfg["model"]

    info_lines = [
        f"Model: SpecFormer (Masked Spectrum Reconstruction)",
        f"",
        f"Architecture:",
        f"  Embed dim:      {model_cfg['embed_dim']}",
        f"  Num layers:     {model_cfg['num_layers']}",
        f"  Num heads:      {model_cfg['num_heads']}",
        f"  MLP expansion:  4x → {model_cfg['embed_dim'] * 4}",
        f"  Max seq len:    {model_cfg['max_len']}",
        f"  Dropout:        {model_cfg['dropout']}",
        f"",
        f"Preprocessing:",
        f"  Slice length:   {model_cfg['slice_section_length']}",
        f"  Slice overlap:  {model_cfg['slice_overlap']}",
        f"  Input dim:      {model_cfg['input_dim']} (= 2 + slice_length)",
        f"",
        f"Masking:",
        f"  Num chunks:     {model_cfg['mask_num_chunks']}",
        f"  Chunk width:    {model_cfg['mask_chunk_width']} tokens",
        f"",
        f"Parameters:",
        f"  Total:          {total_params:,}",
        f"  Trainable:      {trainable_params:,}",
        f"  Size (approx):  {total_params * 4 / 1e6:.1f} MB (fp32)",
        f"",
        f"Training:",
        f"  Optimizer:      AdamW (lr={cfg['optim']['lr']}, wd={cfg['optim']['weight_decay']})",
        f"  Schedule:       Cosine w/ warmup ({cfg['schedule']['warmup_steps']} steps)",
        f"  Max steps:      {cfg['schedule']['max_steps']:,}",
        f"  Epochs:         {cfg['train']['epochs']}",
        f"  Batch size:     {cfg['data']['batch_size']}",
        f"  AMP:            {cfg['train'].get('use_amp', True)}",
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    text = "\n".join(info_lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Figure 10: Model Architecture Summary", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "10_model_summary.png")
    plt.close(fig)
    print("[OK] 10_model_summary.png")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpecFormer Visualization Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HF dataset on disk")
    parser.add_argument("--out_dir", type=str, default="./figures", help="Output directory for figures")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max_embed_samples", type=int, default=500, help="Max samples for t-SNE/PCA")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model, dm, cfg, history = load_model_and_data(args.checkpoint, args.data_path, args.device)
    val_loader = dm.val_dataloader()

    print(f"Collecting samples ...")
    spectra = get_samples(val_loader, n=32, device=args.device)
    print(f"  Got {spectra.shape[0]} samples, spectrum length = {spectra.shape[1]}")

    print("\n=== Generating Figures ===\n")

    plot_loss_curves(history, out_dir)
    plot_mask_illustration(model, spectra, out_dir)
    plot_reconstruction_gallery(model, spectra, out_dir)
    plot_reconstruction_error(model, spectra, out_dir)

    try:
        plot_embedding_space(model, val_loader, out_dir, device=args.device,
                             max_samples=args.max_embed_samples)
    except ImportError:
        print("[SKIP] 5_embedding_space.png — install scikit-learn: pip install scikit-learn")

    try:
        plot_layer_progression(model, spectra, out_dir, device=args.device)
    except ImportError:
        print("[SKIP] 6_layer_progression.png — install scikit-learn")

    plot_attention_heatmaps(model, spectra, out_dir)
    plot_parameter_health(model, out_dir)
    plot_layer_norms(model, spectra, out_dir)
    plot_model_summary(model, cfg, out_dir)

    print(f"\n=== All figures saved to {out_dir.resolve()} ===")


if __name__ == "__main__":
    main()