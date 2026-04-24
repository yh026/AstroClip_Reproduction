import torch
import os
import sys

sys.path.append("/home/users/nus/scratch/AstroCLIP-main/AstroCLIP-main")

from astroclip.data.datamodule import AstroClipDataloader
from astroclip.models.astroclip import AstroClipModel
from astroclip.models.largeimagemodel import vit_large
from astroclip.models.yibinspectrummodel import SpecFormer, SpecFormerConfig

# ================= SAVE =================
SAVE_DIR = "/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/outputs1357/trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "astroclip_best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA =================
dm = AstroClipDataloader()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

# ================= IMAGE ENCODER =================
img_ckpt_path = "/scratch/users/nus/e1554059/astroclip/image_encoder/Final_ViT_model/image_encoder_large_raw_epoch80.pt"
img_ckpt = torch.load(img_ckpt_path, map_location="cpu")

img_cfg = img_ckpt["cfg"]["student"]
crops = img_ckpt["cfg"].get("crops", {})

image_encoder = vit_large(
    img_size=img_cfg.get("img_size", crops.get("global_crops_size", 224)),
    patch_size=img_cfg["patch_size"],
    in_chans=img_cfg.get("in_chans", 3),
    drop_path_rate=img_cfg.get("drop_path_rate", 0.0),
    drop_path_uniform=img_cfg.get("drop_path_uniform", False),
    init_values=img_cfg.get("layerscale", None),
    qkv_bias=img_cfg["qkv_bias"],
    proj_bias=img_cfg["proj_bias"],
    ffn_bias=img_cfg["ffn_bias"],
    num_register_tokens=img_cfg.get("num_register_tokens", 0),
)

image_encoder.load_state_dict(img_ckpt["model"], strict=False)

# ================= SPECTRUM ENCODER =================
spec_ckpt_path = "/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt"
spec_ckpt = torch.load(spec_ckpt_path, map_location="cpu")

spec_cfg_dict = spec_ckpt["config"]
spec_cfg_dict = {k: v for k, v in spec_cfg_dict.items()
                 if k in SpecFormerConfig.__annotations__}

spec_cfg = SpecFormerConfig(**spec_cfg_dict)

spectrum_encoder = SpecFormer(spec_cfg)
spectrum_encoder.load_state_dict(spec_ckpt["model"], strict=False)

# ================= ASTROCLIP =================
model = AstroClipModel(
    image_encoder=image_encoder,
    spectrum_encoder=spectrum_encoder
).to(device)

# freeze backbone
for p in model.image_encoder.parameters():
    p.requires_grad = False
for p in model.spectrum_encoder.parameters():
    p.requires_grad = False

# ================= OPTIM =================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4,
    weight_decay=0.01
)

num_epochs = 100

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

best_val = float("inf")
patience = 5
wait = 0

# ================= TRAIN =================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        spectrum = batch["spectrum"].to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(images, spectrum)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.6f}")

    # ===== VAL =====
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            spectrum = batch["spectrum"].to(device)
            val_loss += model.compute_loss(images, spectrum).item()

    val_loss /= len(val_loader)
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}")

    scheduler.step()

    # ===== SAVE =====
    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        print("Saving best model...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }, MODEL_PATH)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

print("=== Done ===")