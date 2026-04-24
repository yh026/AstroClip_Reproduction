from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image # 图像处理库 Pillow（PIL）中的 Image 模块  Pillow = 用来读/写/处理图片的工具库
# Image 是 Pillow 里最核心的类，用来表示一张图片对象 可以用它打开图片 保存图片 作滤波 转格式
from torchvision import transforms


#! Use
# TO DO 文档最后新增AstroEvalTransform类用于Evaluation
#! 根据论文 默认图像每个像素值已经在输入之前标准化过了 用于标准化的均值和方差是整个图像数据集的 需要确认图像是否已经标准化

class ToRGB:
    """Legacy Survey style arcsinh stretch for g/r/z channels."""

    def __init__(self, scales=None, m: float = 0.03, Q: float = 20.0, bands=None):
        if bands is None:
            bands = ["g", "r", "z"]
        rgb_scales = {
            "u": (2, 1.5),
            "g": (2, 6.0),
            "r": (1, 3.4),
            "i": (0, 1.0),
            "z": (0, 2.2),
        } # 每个元素是 band → (映射到RGB哪个通道, 强度缩放系数)
        # 天文数据不是 RGB 图像 是不同波段（filters） 模型（ViT / CNN）必须吃 3 通道 所以必须：多波段 → 压缩成 RGB
        # 根据天文可视化标准得到缩放规则（行业经验值） 避免某个通道“压死”其他通道
        if scales is not None:
            rgb_scales.update(scales)
        self.rgb_scales = rgb_scales
        self.bands = bands
        self.m = m
        self.Q = Q

    def __call__(self, imgs: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(imgs, torch.Tensor): # imags是单张图片 但是包含3个波段
            imgs = imgs.detach().cpu().numpy()
        if imgs.shape[0] != len(self.bands): # 当前第0维是不是通道数？ len(self.bands)（g r z）= 3
            imgs = np.transpose(imgs, (2, 0, 1)) # 确保图像格式是（C，H, W）而不是(H,W,C)

        I = 0.0
        rgb = np.zeros((3, imgs.shape[1], imgs.shape[2]), dtype=np.float32)
        for img, band in zip(imgs, self.bands): # (g_channel, "g") (r_channel, "r") (z_channel, "z")
            plane, scale = self.rgb_scales[band] # plane是 RGB 的第具体通道  scale强度放大
            img = np.maximum(0.0, img * scale + self.m) # m = 0.03（一个小偏移）防止全黑（避免后面除法炸)+提升暗区域
            # 裁掉负值  来自天文图像噪声 背景扣除可能产生负像素
            rgb[plane] = img # 把当前波段放到 RGB 的某个通道
            I += img #整张图的“亮度（luminance）”或“强度图” I 得到一个“整体亮度参考”
        I /= len(self.bands) #/3 shape (H,W)
        fI = np.arcsinh(self.Q * I) / np.sqrt(self.Q) # arcsinh 天文经典操作 压缩动态范围（亮的不炸，暗的能看见）
        I = np.maximum(I, 1e-6)
        rgb = rgb * (fI / I)[None, :, :] 
        # 用比例调节RGB  RGB 三个通道都会被 同一个比例缩放
        # 这里给二维的（fI/I）最前面加一个通道变成（1，H, W）使得其可以进行ndarray广播 和rgb (3,H,W)每个通道都乘（同一个scale）
        rgb = np.clip(rgb, 0.0, 1.0) # 数值限制在区间：[0, 1] rgb[rgb < 0] = 0 rgb[rgb > 1] = 1 否则超出合法图像范围 → 显示错误
        return np.transpose(rgb, (1, 2, 0)) # 后面用的 augmentation（PIL / numpy）都要求 channel-last 格式 例如Image.fromarray 要求


class RandomGaussianNoise:
    def __init__(self, p: float = 0.5, sigma_min: float = 0.0, sigma_max: float = 0.08):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p: # 返回一个 [0, 1) 之间的均匀分布随机浮点数 以概率 p 执行增强
            return image # 主文件调用该文件 会对该文件设随机种子
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        # 生成一个和图像同 shape 的 高斯噪声（Gaussian noise） 从正态分布 N(mean, std^2) 采样
        return np.clip(image + noise, 0.0, 1.0)  # ndarray


class RandomGaussianBlur: # 在图像上施加模糊扰动，模拟观测分辨率变化（PSF）
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p: # 以概率p执行高斯模糊
            return image
        radius = random.uniform(self.radius_min, self.radius_max)
        pil = Image.fromarray((image * 255).astype(np.uint8)) # PIL 期望的是 [0,255] 的 uint8 图像
        blur = pil.filter(Image.Filter.GaussianBlur(radius=radius)) if hasattr(Image, "Filter") else pil 
        # 如果有 就作高斯模糊 radius越大 模糊程度越大
        arr = np.asarray(blur).astype(np.float32) / 255.0 
        # np.asarray(obj) 把输入“转换成 numpy 数组”，但尽量不复制内存(np.array总是复制) 这里把PIL.Image-> ndarray (dtype = unit8)
        # 转回成float32数据类型 恢复成[0,1]输入格式
        return arr # nadrray


class AstroMultiCropAugmentation:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=144,
        local_crops_size=60,
        blur_probability=0.5,
        noise_probability=0.5,
        use_astro_augmentations=True,
    ):
        self.local_crops_number = int(local_crops_number)
        self.to_rgb = ToRGB() # 类ToRGB的实例ToRGB()作为一个可调用对象 当作一个属性 （这是因为在 Python 里：有 __call__ 的对象 = 可以像函数一样调用）
        self.use_astro_augmentations = use_astro_augmentations

        self.global_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(global_crops_size, scale=tuple(global_crops_scale), antialias=True), #! 额外显式添加antialias说明
                #! 这里是先随机选一块区域（crop 区域大小(面积)比例（与全图相比） ∈ scale (大小随机)  然后再 resize到global_crops_size*global_crops_size 防止锯齿（aliasing）
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        ) # 定义一个“增强流水线”，按顺序对图像执行多个随机变换 Compose把多个 transform 串起来，按顺序执行
        self.local_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(local_crops_size, scale=tuple(local_crops_scale), antialias=True), #! 额外显式添加antialias说明
                # 随机选一个区域（crop） 区域大小(面积)比例（与全图相比） ∈ scale  resize 到固定尺寸（local_crops_size）  防止锯齿（aliasing）
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.noise = RandomGaussianNoise(p=noise_probability) #返回array
        self.blur = RandomGaussianBlur(p=blur_probability) #返回array

    def _postprocess(self, crop: torch.Tensor) -> torch.Tensor:
        if self.use_astro_augmentations: # multi-band → RGB（用 arcsinh stretch）
            rgb = self.to_rgb(crop) # equivalent to self.to_rgb.__call__(x) # 输出 (H,W,3)
        else: # 已经是 RGB
            arr = crop.detach().cpu().numpy() # 转化为numpy 因为是torch来的（__call__的代码中g1可见） 应该是这个格式(C,H,W)
            if arr.shape[0] == 3: 
                rgb = np.transpose(np.clip(arr, 0.0, 1.0), (1, 2, 0)) # (C,H,W) → (H,W,C)
            else: # arr 不是 3 通道 多波段图像（比如 g,r,z,...）
                rgb = self.to_rgb(arr[:3]) # 取前3个波段 → 转RGB （H,W,C）
        rgb = self.blur(rgb) # 模拟分辨率变化
        rgb = self.noise(rgb) # 模拟观测噪声
        return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() # 转换回tensor (H,W,C)-> (C,H,W) 

    def __call__(self, image: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        g1 = self._postprocess(self.global_aug(image)) #注意g1 g2是用同一个图像image生成的
        g2 = self._postprocess(self.global_aug(image)) #! 先裁剪 再转化成rgb(符合github)
        locals_ = [self._postprocess(self.local_aug(image)) for _ in range(self.local_crops_number)] # 局部图像也是
        return {
            "global_crops": [g1, g2], # 每个元素是矩阵图像
            "global_crops_teacher": [g1, g2],
            "local_crops": locals_,
        }

 # TO DO 不带随机增强、只做固定必要预处理的 单图transform
# 训练时原本就是通过 ToRGB + 后处理 把多波段图送进 ViT，所以这样做是最自然的测试入口
class AstroEvalTransform:
    """
    Deterministic transform for evaluation / embedding export.
    Output: a single CHW float tensor.
    """
    def __init__(self, image_size=144, use_astro_augmentations=True): 
        self.image_size = image_size
        self.use_astro_augmentations = use_astro_augmentations
        self.to_rgb = ToRGB()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: expected CHW
        if self.use_astro_augmentations:
            rgb = self.to_rgb(image)  # HWC in [0,1]
        else:
            arr = image.detach().cpu().numpy()
            if arr.shape[0] == 3:
                rgb = np.transpose(np.clip(arr, 0.0, 1.0), (1, 2, 0))
            else:
                rgb = self.to_rgb(arr[:3])

        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()  # CHW

        # deterministic resize to training global size
        # 验证集必须和训练时的输入分布一致（尤其是尺寸） 模型是“对固定尺寸学习的”
        tensor = transforms.functional.resize( # 对图像进行尺寸调整（resize）
            # resize 本质: 不是简单拉伸，而是“插值（interpolation）” 
            # 原图像是一个离散网格 我们要在新网格上“重新采样像素值” +插值
            tensor, # image (C,H,W)
            [self.image_size, self.image_size], 
            # size 1. 写成 tuple/list 直接 resize 到指定尺寸
            # 写成 int 指定 短边，长边按比例缩放
            antialias=True,
        )
        return tensor

        '''
        关于evaluation时候的图像resize得到的尺寸:
        模型真正学的是 global view → eval 就必须喂 global view

        不是原始图像的尺寸 因为训练前用的已经是原始图像transform之后的结果了
        也不是local尺寸 那只是训练时的辅助视角 算是training trick
        global crop才是模型学习的“主语义空间” teacher只看global eval本质更接近teacher forward
        embedding空间也是基于global crop分布学出来的
        导出最后的embedding也是用teacher backbone 是EMA累积得到的 更稳定 更适合作最终表征输出
        '''