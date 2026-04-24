# import torch
# from astroclip.models.astroclip import AstroClipModel, ImageHead, SpectrumHead

# ckpt_img = torch.load("/scratch/users/nus/e1554059/astroclip/image_encoder/Final_ViT_model/image_encoder_small_std_epoch400.pt", map_location="cpu")
# ckpt_spec = torch.load("/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt", map_location="cpu")["model"]

# image_encoder = ImageHead(
#     config="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/astroclip/astrodino/config.yaml",
#     model_weights="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/astrodino.ckpt",
#     save_directory="./tmp"
# )

# spectrum_encoder = SpectrumHead(
#     model_path="/home/users/nus/e1632476/scratch/AstroCLIP-main/AstroCLIP-main/pretrained/specformer.ckpt"
# )

# model = AstroClipModel(image_encoder=image_encoder, spectrum_encoder=spectrum_encoder)

# missing_img, unexpected_img = model.image_encoder.load_state_dict(ckpt_img, strict=False)
# missing_spec, unexpected_spec = model.spectrum_encoder.load_state_dict(ckpt_spec, strict=False)

# print(len(missing_img), len(unexpected_img))
# print(len(missing_spec), len(unexpected_spec))

# import torch

# ckpt = torch.load("/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt", map_location="cpu")

# print(ckpt.keys())                 # 看外层结构
# print(ckpt["model"].keys())       # 看真实权重结构（state_dict）

#======================================================================================================================================

# import torch
# from astroclip.models.yibinspectrummodel import SpecFormer, SpecFormerConfig

# ckpt = torch.load("/home/users/nus/e1538405/DSA5204/astroclip_spectrum_minimal/outputs_specformer/best.pt", map_location="cpu")

# cfg_dict = ckpt["config"]
# cfg_dict = {k: v for k, v in cfg_dict.items() if k in SpecFormerConfig.__annotations__}

# cfg = SpecFormerConfig(**cfg_dict)

# model = SpecFormer(cfg)
# model.load_state_dict(ckpt["model"], strict=False)

# print("load done")

#======================================================================================================================================

import torch
import sys

sys.path.append("/home/users/nus/scratch/AstroCLIP-main/AstroCLIP-main")

from astroclip.models.largeimagemodel import vit_large

ckpt_path = "/scratch/users/nus/e1554059/astroclip/image_encoder/Final_ViT_model/image_encoder_large_raw_epoch80.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")

cfg = ckpt["cfg"]["student"]
crops = ckpt["cfg"].get("crops", {})

# -------- 安全解析（避免 KeyError）--------
img_size = cfg.get("img_size", crops.get("global_crops_size", 224))
patch_size = cfg.get("patch_size", None)
if patch_size is None:
    raise ValueError("patch_size not found in checkpoint cfg")

in_chans = cfg.get("in_chans", 3)
drop_path_rate = cfg.get("drop_path_rate", 0.0)
drop_path_uniform = cfg.get("drop_path_uniform", False)
layerscale = cfg.get("layerscale", None)
num_register_tokens = cfg.get("num_register_tokens", 0)

# -------- 构建模型（严格对齐 ckpt）--------
model = vit_large(
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

# -------- 加载权重 --------
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

print("loaded")
print("missing:", len(missing))
print("unexpected:", len(unexpected))