"""
Redshift Estimation from your trained AstroCLIP (trainer3)
==========================================================

All astroclip imports use importlib with full file paths.
No need to install or write to the astroclip package.

Usage:
    python redshift_estimation.py
"""

import os
import sys
import types
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
ASTROCLIP_ROOT = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main"
ASTROCLIP_PKG  = os.path.join(ASTROCLIP_ROOT, "astroclip")

CLIP_CKPT     = os.path.join(ASTROCLIP_ROOT, "outputs1234/trained_models/astroclip_trainer3_best.pt")
SPECTRUM_CKPT = "/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt"
IMAGE_CKPT    = "/scratch/users/nus/e1554059/astroclip/image_encoder/Final_ViT_model/image_encoder_large_raw_epoch80.pt"

DATASET_DIR   = "/scratch/users/nus/e0492520/astroclip_pipeline/exported/astroclip_dataset"

OUTPUT_DIR    = "./redshift_results_f"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# ============================================================
# IMPORT HELPER: load modules by full file path
# ============================================================

def _load_module(name, filepath):
    """Load a single .py file as a named module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_package(pkg_name, pkg_dir):
    """Register a package namespace in sys.modules (like a virtual __init__.py)."""
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg


def setup_astroclip_imports():
    """
    Register the astroclip package hierarchy and load ONLY the model modules.
    No datamodule.py — we load the dataset directly from disk.

    Dependency chain:
      largeimagemodel.py  →  astroclip.models.layers  (Mlp, PatchEmbed, Block)
      yibinspectrummodel.py  →  astroclip.yibinmodules  (LayerNorm, TransformerBlock, init_linear_by_depth)
    """
    models_dir = os.path.join(ASTROCLIP_PKG, "models")

    # 1. Register package namespaces
    _ensure_package("astroclip",        ASTROCLIP_PKG)
    _ensure_package("astroclip.models", models_dir)

    # 2. Load dependency modules FIRST (order matters)
    _load_module("astroclip.yibinmodules",
                 os.path.join(ASTROCLIP_PKG, "yibinmodules.py"))

    _load_module("astroclip.models.layers",
                 os.path.join(models_dir, "layers.py"))

    # 3. Load the modules we actually need
    spec_mod = _load_module("astroclip.models.yibinspectrummodel",
                            os.path.join(models_dir, "yibinspectrummodel.py"))

    img_mod = _load_module("astroclip.models.largeimagemodel",
                           os.path.join(models_dir, "largeimagemodel.py"))

    return img_mod, spec_mod


# ============================================================
# STEP 1: Load all modules
# ============================================================

print("Setting up imports...")
img_mod, spec_mod = setup_astroclip_imports()

# Pull out the classes/functions we need
vit_large       = img_mod.vit_large
SpecFormer      = spec_mod.SpecFormer
SpecFormerConfig = spec_mod.SpecFormerConfig


# ============================================================
# STEP 2: Dataset loading
# ============================================================

def get_dataloaders():
    """
    Load dataset from HuggingFace Arrow format using the `datasets` library.
    Dataset structure: astroclip_dataset/{dataset_dict.json, train/, test/}
    Batch keys: ['image', 'spectrum', 'redshift']
    """
    from datasets import load_from_disk
    from torch.utils.data import Dataset

    class HFDataset(Dataset):
        def __init__(self, hf_split):
            self.data = hf_split
            print(f"    Loaded {len(self.data)} samples, columns: {self.data.column_names}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data[idx]
            # Image is stored as (H, W, C), PyTorch needs (C, H, W)
            image = torch.tensor(row["image"], dtype=torch.float32)
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1)
            return {
                "image":    image,
                "spectrum": torch.tensor(row["spectrum"], dtype=torch.float32),
                "redshift": torch.tensor(row["redshift"], dtype=torch.float32),
            }

    print(f"Loading dataset from: {DATASET_DIR}")
    ds = load_from_disk(DATASET_DIR)

    print("  Train split:")
    train_ds = HFDataset(ds["train"])
    print("  Test split:")
    test_ds  = HFDataset(ds["test"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# STEP 3: Rebuild and load the full AstroCLIP model
#   Updated: CrossAttentionHead (512-d), patch tokens
# ============================================================

class CrossAttentionHead(nn.Module):
    def __init__(self, embed_dim=512, n_head=4, token_dim=1024, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.kv_proj = nn.Linear(token_dim, embed_dim * 2, bias=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_head,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.xavier_uniform_(self.kv_proj.weight)

    def forward(self, tokens):
        B = tokens.size(0)
        kv = self.kv_proj(tokens)
        k, v = kv.chunk(2, dim=-1)
        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(1)
        x = self.norm1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ImageBackboneWrapper(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt["cfg"]["student"]
        crops = ckpt["cfg"].get("crops", {})
        self.backbone = vit_large(
            img_size=cfg.get("img_size", crops.get("global_crops_size", 224)),
            patch_size=cfg["patch_size"],
            in_chans=cfg.get("in_chans", 3),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            drop_path_uniform=cfg.get("drop_path_uniform", False),
            init_values=cfg.get("layerscale", None),
            qkv_bias=cfg["qkv_bias"],
            proj_bias=cfg["proj_bias"],
            ffn_bias=cfg["ffn_bias"],
            num_register_tokens=cfg.get("num_register_tokens", 0),
        )
        self.backbone.load_state_dict(ckpt["model"], strict=False)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        ret = self.backbone.forward_features(x)
        return ret["x_norm_patchtokens"]  # (B, N, 1024)


class SpectrumBackboneWrapper(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = {k: v for k, v in ckpt["config"].items()
                    if k in SpecFormerConfig.__annotations__}
        self.backbone = SpecFormer(SpecFormerConfig(**cfg_dict))
        self.backbone.load_state_dict(ckpt["model"], strict=False)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs["embedding"]  # (B, seq_len, 768)


class AstroClipModelV3(nn.Module):
    IMG_TOKEN_DIM  = 1024
    SPEC_TOKEN_DIM = 768
    EMBED_DIM      = 512

    def __init__(self, image_backbone, spectrum_backbone):
        super().__init__()
        self.image_encoder    = image_backbone
        self.spectrum_encoder = spectrum_backbone
        self.img_head  = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.IMG_TOKEN_DIM)
        self.spec_head = CrossAttentionHead(
            embed_dim=self.EMBED_DIM, n_head=4, token_dim=self.SPEC_TOKEN_DIM)
        self.criterion = None  # not needed for inference

    def encode_image(self, x):
        with torch.no_grad():
            tokens = self.image_encoder(x)       # (B, N, 1024)
        feat = self.img_head(tokens)              # (B, 512)
        return F.normalize(feat, dim=-1)

    def encode_spectrum(self, x):
        with torch.no_grad():
            tokens = self.spectrum_encoder(x)    # (B, S, 768)
        feat = self.spec_head(tokens)             # (B, 512)
        return F.normalize(feat, dim=-1)


def load_astroclip():
    print("Loading image backbone (vit_large → patch tokens)...")
    img_bb = ImageBackboneWrapper(IMAGE_CKPT)
    print("Loading spectrum backbone (SpecFormer → token sequence)...")
    spec_bb = SpectrumBackboneWrapper(SPECTRUM_CKPT)
    model = AstroClipModelV3(img_bb, spec_bb)

    print(f"Loading CLIP weights from {CLIP_CKPT}...")
    ckpt = torch.load(CLIP_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  Loaded (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f})")
    return model.to(DEVICE).eval()


# ============================================================
# STEP 4: Extract embeddings
# ============================================================

def extract_embeddings(model, loader, modality="image"):
    """Extract embeddings from a single dataloader split."""
    all_emb, all_z = [], []
    print(f"  Extracting {modality} ({len(loader)} batches)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if modality == "image":
                z = model.encode_image(batch["image"].to(DEVICE))
            else:
                z = model.encode_spectrum(batch["spectrum"].to(DEVICE))
            all_emb.append(z.cpu().numpy())
            all_z.append(batch["redshift"].cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(loader)}")
    emb = np.concatenate(all_emb)
    red = np.concatenate(all_z)
    print(f"  Done: {emb.shape[0]} samples, dim={emb.shape[1]}")
    return emb, red


# ============================================================
# STEP 5: k-NN (zero-shot)
# ============================================================

def run_knn(z_tr, y_tr, z_te, k=16):
    print(f"\n  k-NN (k={k})...")
    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(z_tr, y_tr)
    return knn.predict(z_te)


# ============================================================
# STEP 6: MLP (few-shot)
# ============================================================

class RedshiftMLP(nn.Module):
    def __init__(self, d_in=1024, d_hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_mlp(z_tr, y_tr, z_te, hid=32, lr=1e-3, epochs=200, bs=256):
    print(f"\n  MLP (hidden={hid}, epochs={epochs})...")
    ds = TensorDataset(torch.tensor(z_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    mlp = RedshiftMLP(d_in=z_tr.shape[1], d_hid=hid).to(DEVICE)
    opt = optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss, best_state = float("inf"), None
    mlp.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(mlp(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        if (ep + 1) % 50 == 0:
            print(f"    Epoch {ep+1}/{epochs} — MSE = {avg:.6f}")

    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        return mlp(torch.tensor(z_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()


# ============================================================
# STEP 7: Plotting (Fig. 3 style)
# ============================================================

def plot_results(y_true, y_pred, r2, res, title, path):
    """Plot redshift scatter + residuals using the authors' seaborn style."""

    data_lower_lim = 0.0
    data_upper_lim = 0.6

    fig, ax = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={"height_ratios": [2, 1]})

    # --- Top panel: prediction vs true (authors' style) ---
    sns.scatterplot(ax=ax[0], x=y_true, y=y_pred, s=5, color=".15", alpha=0.3)
    sns.histplot(ax=ax[0], x=y_true, y=y_pred, bins=50, pthresh=0.1, cmap="mako")
    sns.kdeplot(ax=ax[0], x=y_true, y=y_pred, levels=5, color="k", linewidths=1)

    ax[0].plot([data_lower_lim, data_upper_lim], [data_lower_lim, data_upper_lim],
               "--", linewidth=1.5, alpha=0.5, color="grey")

    ax[0].set_xlim(data_lower_lim, data_upper_lim)
    ax[0].set_ylim(data_lower_lim, data_upper_lim)
    ax[0].set_xlabel(r"$Z_{\rm true}$", fontsize=20)
    ax[0].set_ylabel(r"$Z_{\rm pred}$", fontsize=20)
    ax[0].set_title(title, fontsize=22)
    ax[0].text(0.9, 0.1, f"$R^2$ score: {r2:.2f}",
               horizontalalignment="right", verticalalignment="top",
               fontsize=20, transform=ax[0].transAxes)

    # --- Bottom panel: residuals (authors' style) ---
    x = y_true
    y = res

    bins = np.linspace(data_lower_lim, data_upper_lim * 1.05, 20)
    x_binned = np.digitize(x, bins)
    y_std = [y[x_binned == i].std() if (x_binned == i).sum() > 5 else np.nan
             for i in range(1, len(bins))]

    sns.scatterplot(ax=ax[1], x=x, y=y, s=2, alpha=0.3, color="black")
    sns.lineplot(ax=ax[1], x=bins[:-1], y=y_std, color="r", label="std")

    ax[1].axhline(0, color="grey", linewidth=1.5, alpha=0.5, linestyle="--")
    ax[1].set_xlim(data_lower_lim, data_upper_lim)
    ax[1].set_ylim(-data_upper_lim / 2, data_upper_lim / 2)
    ax[1].set_xlabel(r"$Z_{\rm true}$", fontsize=20)
    ax[1].set_ylabel(r"$(Z_{\rm true}-Z_{\rm pred})/(1+Z_{\rm true})$", fontsize=20)
    ax[1].legend(fontsize=15, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def evaluate_redshift(z_tr_emb, z_te_emb, y_tr_raw, y_te_raw, label, output_dir):
    """
    Authors' methodology with z<0.6 filtering:
    1. Filter train and test to z < 0.6
    2. StandardScaler on embeddings (fit on filtered train)
    3. Z-score redshifts (fit on filtered train)
    4. Train k-NN/MLP, inverse-transform predictions
    5. Report R² on filtered test set
    """
    Z_CUT = 0.6

    # Filter to z < 0.6
    train_mask = y_tr_raw < Z_CUT
    test_mask  = y_te_raw < Z_CUT
    emb_tr_f = z_tr_emb[train_mask]
    emb_te_f = z_te_emb[test_mask]
    y_tr_f   = y_tr_raw[train_mask]
    y_te_f   = y_te_raw[test_mask]

    print(f"  Filtered: train {train_mask.sum()}/{len(y_tr_raw)}, test {test_mask.sum()}/{len(y_te_raw)} (z < {Z_CUT})")

    # StandardScaler on embeddings
    scaler = StandardScaler().fit(emb_tr_f)
    X_tr = scaler.transform(emb_tr_f)
    X_te = scaler.transform(emb_te_f)

    # Z-score redshifts
    z_mean, z_std = y_tr_f.mean(), y_tr_f.std()
    y_tr_scaled = (y_tr_f - z_mean) / z_std

    results = {}
    for method_name, run_fn in [("k-NN", lambda: run_knn(X_tr, y_tr_scaled, X_te)),
                                 ("MLP",  lambda: run_mlp(X_tr, y_tr_scaled, X_te))]:
        yp_scaled = run_fn()
        yp = yp_scaled * z_std + z_mean

        r2 = r2_score(y_te_f, yp)
        res = (y_te_f - yp) / (1 + y_te_f)

        print(f"  {method_name} → R²(z<{Z_CUT}) = {r2:.4f}  |  Res.std = {res.std():.4f}")

        plot_results(y_te_f, yp, r2, res, f"{label} — {method_name}",
                     os.path.join(output_dir, f"redshift_{method_name.lower()}_{label.lower()}.png"))
        results[method_name] = (r2, res)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_DIR}\n")

    # 1. Load model
    model = load_astroclip()

    # 2. Get data loaders (already split into train/test)
    train_loader, test_loader = get_dataloaders()

    # ==========================================
    # IMAGE-BASED REDSHIFT
    # ==========================================
    print("\n" + "=" * 55)
    print("  IMAGE-BASED REDSHIFT ESTIMATION")
    print("=" * 55)
    emb_img_tr, z_train = extract_embeddings(model, train_loader, "image")
    emb_img_te, z_test  = extract_embeddings(model, test_loader,  "image")
    np.save(os.path.join(OUTPUT_DIR, "clip_image_emb_train.npy"), emb_img_tr)
    np.save(os.path.join(OUTPUT_DIR, "clip_image_emb_test.npy"),  emb_img_te)
    np.save(os.path.join(OUTPUT_DIR, "z_train.npy"), z_train)
    np.save(os.path.join(OUTPUT_DIR, "z_test.npy"),  z_test)

    print(f"\n  Train: {len(z_train)} | Test: {len(z_test)}")
    img_results = evaluate_redshift(
        emb_img_tr, emb_img_te, z_train, z_test, "Image", OUTPUT_DIR)

    # ==========================================
    # SPECTRUM-BASED REDSHIFT
    # ==========================================
    print("\n" + "=" * 55)
    print("  SPECTRUM-BASED REDSHIFT ESTIMATION")
    print("=" * 55)
    emb_sp_tr, _ = extract_embeddings(model, train_loader, "spectrum")
    emb_sp_te, _ = extract_embeddings(model, test_loader,  "spectrum")
    np.save(os.path.join(OUTPUT_DIR, "clip_spec_emb_train.npy"), emb_sp_tr)
    np.save(os.path.join(OUTPUT_DIR, "clip_spec_emb_test.npy"),  emb_sp_te)

    spec_results = evaluate_redshift(
        emb_sp_tr, emb_sp_te, z_train, z_test, "Spectrum", OUTPUT_DIR)

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  V3 RESULTS (z < 0.6 only)")
    print(f"{'='*60}")
    print(f"  {'Source':<10} {'Method':<20} {'R²':>8}  {'Res.Std':>8}")
    print(f"  {'-'*50}")
    for source, results in [("Image", img_results), ("Spectrum", spec_results)]:
        for method, (r2, res) in results.items():
            print(f"  {source:<10} {method:<20} {r2:>8.4f}  {res.std():>8.4f}")
    print(f"\n  Paper reference:")
    print(f"  {'Image':<10} {'k-NN':<20} {'0.79':>8}")
    print(f"  {'Image':<10} {'MLP':<20} {'0.78':>8}")
    print(f"  {'Spectrum':<10} {'MLP':<20} {'0.98':>8}")
    print(f"{'='*60}")
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()