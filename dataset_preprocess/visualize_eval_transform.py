import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk

# from load_data_subset import load_astroclip_subset
import augmentations


NUM_IMAGES = 8
IMAGE_SIZE = 144


def main():
    # 读取数据
    # train, test = load_astroclip_subset()
    dset = load_from_disk("/scratch/users/nus/e0492520/astroclip_pipeline/exported/astroclip_dataset")
    train = dset["train"]
    test = dset["test"]

    # 初始化 transform
    image_transform = augmentations.AstroEvalTransform(
        image_size=IMAGE_SIZE,
        use_astro_augmentations=True,
    )

    raw_images = []
    transformed_images = []

    # 取前 8 张
    for i in range(NUM_IMAGES):
        sample = train[i]

        image = torch.tensor(np.array(sample["image"]), dtype=torch.float32)

        # 转 CHW
        if image.ndim == 3 and image.shape[-1] == 3:
            image_chw = image.permute(2, 0, 1)
            raw_hwc = image.numpy()
        elif image.ndim == 3 and image.shape[0] == 3:
            image_chw = image
            raw_hwc = image.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        raw_images.append(raw_hwc)

        # transform 后图像
        image_transformed = image_transform(image_chw)
        transformed_hwc = image_transformed.permute(1, 2, 0).numpy()

        transformed_images.append(transformed_hwc)

    # ----------------------------
    # 合并画图（2行8列）
    # ----------------------------
    fig, axes = plt.subplots(2, NUM_IMAGES, figsize=(16, 5))
    fig.suptitle("Raw vs AstroEvalTransform", fontsize=16)

    # 第一行：原始图像
    for i in range(NUM_IMAGES):
        axes[0, i].imshow(raw_images[i])
        axes[0, i].set_title(f"Raw {i}")
        axes[0, i].axis("off")

    # 第二行：transform 后
    for i in range(NUM_IMAGES):
        axes[1, i].imshow(transformed_images[i])
        axes[1, i].set_title(f"Trans {i}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Raw", fontsize=12)
    axes[1, 0].set_ylabel("After Transform", fontsize=12)

    plt.tight_layout()
    plt.savefig("comparison_raw_vs_transform.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()