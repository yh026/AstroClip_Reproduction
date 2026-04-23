from datasets import load_from_disk

import os
import sys
import numpy as np


EXPORT_DIR = "/scratch/users/nus/e0492520/astroclip_pipeline/exported/astroclip_subset_processed"
NORMALIZED_DIR = "/scratch/users/nus/e0492520/astroclip_pipeline/exported/astroclip_dataset"

WORK_ROOT = "/scratch/users/nus/e0492520/astroclip_pipeline"
CACHE_ROOT = os.path.join(WORK_ROOT, "cache")
EXPORT_ROOT = os.path.join(WORK_ROOT, "exported")

PROCESSED_TRAIN_CACHE = os.path.join(CACHE_ROOT, "train_processed.arrow")
PROCESSED_TEST_CACHE = os.path.join(CACHE_ROOT, "test_processed.arrow")

EXPORT_DIR = os.path.join(EXPORT_ROOT, "astroclip_subset_processed")

SEED = 42
EPS = 1e-8
NUM_PROC = int(os.environ.get("PBS_NCPUS", "4"))
BATCH_SIZE = 256

def compute_image_stats_from_processed(ds, split="train", batch_size=256):
    import numpy as np

    dataset = ds[split]
    n = len(dataset)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]

        imgs = np.stack(batch["image"], axis=0).astype(np.float32)  # (B, 144, 144, 3)

        channel_sum += imgs.sum(axis=(0, 1, 2))
        channel_sum_sq += (imgs ** 2).sum(axis=(0, 1, 2))
        total_pixels += imgs.shape[0] * imgs.shape[1] * imgs.shape[2]

    mean = channel_sum / total_pixels
    var = channel_sum_sq / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    std = np.where(std < EPS, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)

def make_image_norm_transform(mean, std):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def transform(batch):
        images = []

        for img in batch["image"]:
            img = np.asarray(img, dtype=np.float32)
            img = (img - mean) / (std + EPS)
            images.append(img.astype(np.float32))

        batch["image"] = images
        return batch

    return transform

def normalize_images_only(ds):
    print("Computing image stats from processed TRAIN set...")
    mean, std = compute_image_stats_from_processed(ds, "train")

    print("Image mean:", mean)
    print("Image std :", std)

    transform = make_image_norm_transform(mean, std)

    print("Applying normalization...")

    ds["train"] = ds["train"].map(
        transform,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        desc="Normalizing TRAIN images",
    )

    ds["test"] = ds["test"].map(
        transform,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        desc="Normalizing TEST images",
    )

    return ds

if __name__ == "__main__":
    ds = load_from_disk(EXPORT_DIR)
    ds = normalize_images_only(ds) 
    
    ds.save_to_disk(NORMALIZED_DIR)
    
    # verify
    mean, std = compute_image_stats_from_processed(ds, "train")
    print(mean, std)