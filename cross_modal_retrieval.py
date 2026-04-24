"""
evaluate_retrieval.py
跨模态对齐评估脚本：Image-to-Spectrum & Spectrum-to-Image Retrieval
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── 1. 路径与依赖导入 ────────────────────────────────────────────────────────
# 导入你 trainer3.py 中的环境变量和网络结构
try:
    from trainer3 import (
        AstroClipModelV3, 
        ImageBackboneWrapper, 
        SpectrumBackboneWrapper,
        IMAGE_CKPT, 
        SPECTRUM_CKPT, 
        MODEL_PATH,
        ASTROCLIP_ROOT
    )
except ImportError:
    raise ImportError("❌ 找不到 trainer3.py，请确保该脚本与 trainer3.py 放在同一目录下！")

# 确保 AstroCLIP 库在环境变量中
sys.path.insert(0, ASTROCLIP_ROOT)
from astroclip.data.datamodule import AstroClipDataloader

# ── 2. 评估核心逻辑 ────────────────────────────────────────────────────────
def evaluate_cross_modal_retrieval(model, dataloader, device):
    """
    计算跨模态检索的 Recall@1, Recall@5, Recall@10
    """
    model.eval()
    
    all_image_embeds = []
    all_spec_embeds = []
    
    print("\n🔍 第一阶段: 正在提取全局跨模态特征 (Extracting embeddings)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Feature Extraction"):
            # 取出数据并推送到设备
            images = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)
            
            # 【核心契合点】直接使用你 AstroClipModelV3 中定义的 forward 方法
            # forward 会自动依次经过 backbone -> proj layer，并输出 1024 维特征
            img_features, spec_features = model(images, spectra)
            
            # 【必须步骤】L2 归一化：为了后续可以直接使用内积代表余弦相似度
            img_features = F.normalize(img_features, dim=-1)
            spec_features = F.normalize(spec_features, dim=-1)
            
            # 将特征推送到 CPU 暂存池，避免 Dataloader 数据过多撑爆显存
            all_image_embeds.append(img_features.cpu())
            all_spec_embeds.append(spec_features.cpu())
            
    # 拼接全量数据集特征
    Z_im = torch.cat(all_image_embeds, dim=0)  # Shape: (N, 1024)
    Z_sp = torch.cat(all_spec_embeds, dim=0)   # Shape: (N, 1024)
    
    N = Z_im.shape[0]
    print(f"✅ 提取完成！测试集样本总数 (Total evaluate samples): {N}")
    
    # ── 计算相似度矩阵 ──
    print("\n🧮 第二阶段: 计算全局 N x N 相似度矩阵...")
    
    # 【显存保护机制】如果样本超过 2 万，强行在 GPU 算 N x N 矩阵可能会 OOM。
    # 这里做了一个动态分配机制。
    calc_device = device if N <= 25000 else torch.device("cpu")
    if calc_device.type == "cpu" and device.type == "cuda":
        print("⚠️ 样本量较大，为了防止显存溢出 (OOM)，相似度矩阵计算将回退至 CPU 进行。")
        
    Z_im = Z_im.to(calc_device)
    Z_sp = Z_sp.to(calc_device)
    
    # 计算余弦相似度矩阵 (N x N)
    similarity_matrix = Z_im @ Z_sp.T  
    
    K_vals = [1, 5, 10]
    i2s_recall = {k: 0.0 for k in K_vals}
    s2i_recall = {k: 0.0 for k in K_vals}
    
    # 真实匹配的索引就是对角线 0, 1, 2... N-1
    targets = torch.arange(N, device=calc_device).unsqueeze(1)
    
    # ── 执行检索与统计 ──
    print("📊 第三阶段: 统计 Top-K 命中率 (Recall@K)...")
    
    # Image-to-Spectrum (I2S): 每行找最大的 K 个值
    _, i2s_topk_indices = similarity_matrix.topk(max(K_vals), dim=1, largest=True, sorted=True)
    
    # Spectrum-to-Image (S2I): 矩阵转置后，按列找最大的 K 个值
    _, s2i_topk_indices = similarity_matrix.t().topk(max(K_vals), dim=1, largest=True, sorted=True)
    
    for k in K_vals:
        # 统计 targets 是否在这前 K 个预测索引中
        i2s_correct = (i2s_topk_indices[:, :k] == targets).any(dim=1).float().sum().item()
        s2i_correct = (s2i_topk_indices[:, :k] == targets).any(dim=1).float().sum().item()
        
        i2s_recall[k] = (i2s_correct / N) * 100
        s2i_recall[k] = (s2i_correct / N) * 100
        
    # ── 打印最终报告 ──
    print("\n" + "═"*50)
    print("🏆 跨模态检索最终评估报告 (Cross-Modal Retrieval) ")
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
    # 动态检测硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动评估脚本 | 当前运行设备 (Device): {device}")
    
    # 构建 DataLoader (优先使用验证集或测试集)
    dm = AstroClipDataloader()
    # 对于检索任务，通常使用验证集(val)或者测试集(test)进行无偏评估
    test_loader = dm.val_dataloader() 
    
    # 加载网络结构 (直接复用 trainer3 的包装类)
    print("\n📦 正在构建网络结构...")
    image_backbone    = ImageBackboneWrapper(IMAGE_CKPT)
    spectrum_backbone = SpectrumBackboneWrapper(SPECTRUM_CKPT)
    model = AstroClipModelV3(image_backbone, spectrum_backbone)
    
    # 加载最佳模型权重
    print(f"📂 正在加载训练好的权重: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到权重文件 {MODEL_PATH}，请确认 trainer3.py 是否已经训练完毕！")
        
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    # 过滤可能存在的多余前缀并加载
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("✅ 权重加载成功！")
    
    # 开始评估
    evaluate_cross_modal_retrieval(model, test_loader, device)

if __name__ == "__main__":
    main()