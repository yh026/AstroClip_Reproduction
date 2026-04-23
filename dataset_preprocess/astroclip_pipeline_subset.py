import os
import sys
import numpy as np
from datasets import Dataset, DatasetDict, Features, Array3D, Array2D, Value

# ===== 关键：加入你的代码路径 =====
sys.path.append("/scratch/users/nus/e1553819/astroclip/code")
from load_data_subset import load_astroclip_subset

# ===== 路径配置 =====
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

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(EXPORT_ROOT, exist_ok=True)


# ===== preprocessing =====
def center_crop(image):
    image = np.asarray(image, dtype=np.float32)
    return image[4:148, 4:148, :]  # 152 -> 144


def process_spectrum(spectrum):
    spec = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    mu = spec.mean()
    sigma = spec.std()

    z = (spec - mu) / (sigma + EPS)
    out = np.concatenate([z, np.array([mu, sigma], dtype=np.float32)])
    return out.reshape(-1, 1)


def transform_batch(batch):
    images = []
    spectra = []
    redshifts = []
    targetids = []

    for img, spec, z, tid in zip(
        batch["image"], batch["spectrum"], batch["redshift"], batch["targetid"]
    ):
        images.append(center_crop(img))
        spectra.append(process_spectrum(spec))
        redshifts.append(np.float32(z))
        targetids.append(int(tid))

    return {
        "image": images,
        "spectrum": spectra,
        "redshift": redshifts,
        "targetid": targetids,
    }


def get_features():
    return Features({
        "image": Array3D(shape=(144, 144, 3), dtype="float32"),
        "spectrum": Array2D(shape=(7783, 1), dtype="float32"),
        "redshift": Value("float32"),
        "targetid": Value("int64"),
    })


# ===== 核心 pipeline =====
def build_dataset(force_rebuild=False):
    print("Loading subset dataset...")

    train_raw, test_raw = load_astroclip_subset()

    # 转成 HF Dataset（如果还不是的话）
    if not isinstance(train_raw, Dataset):
        train_raw = Dataset.from_dict(train_raw)
    if not isinstance(test_raw, Dataset):
        test_raw = Dataset.from_dict(test_raw)

    print("Train size:", len(train_raw))
    print("Test size :", len(test_raw))

    # ===== map（带 cache）=====
    train_ds = train_raw.map(
        transform_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        features=get_features(),
        cache_file_name=PROCESSED_TRAIN_CACHE,
        load_from_cache_file=True,
        desc="Processing TRAIN",
    )

    test_ds = test_raw.map(
        transform_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        features=get_features(),
        cache_file_name=PROCESSED_TEST_CACHE,
        load_from_cache_file=True,
        desc="Processing TEST",
    )

    ds = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    return ds


# ===== 导出（只在需要共享时用）=====
def export_dataset(force_overwrite=False):
    ds = build_dataset()

    if os.path.exists(EXPORT_DIR):
        if force_overwrite:
            import shutil
            shutil.rmtree(EXPORT_DIR)
        else:
            raise ValueError(f"{EXPORT_DIR} already exists")

    print("Saving dataset (this will be slow)...")
    ds.save_to_disk(EXPORT_DIR)

    print("Saved to:", EXPORT_DIR)
    
# ===== image standardized ===== #

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


# ===== 验证 =====
def verify(ds, n=3):
    print(ds)
    print(ds["train"].features)

    for i in range(n):
        ex = ds["train"][i]
        img = np.array(ex["image"])
        spec = np.array(ex["spectrum"]).reshape(-1)

        main = spec[:-2]
        mu = spec[-2]
        sigma = spec[-1]

        print(f"\nSample {i}")
        print("image:", img.shape)
        print("spec :", spec.shape)
        print("mean :", main.mean())
        print("std  :", main.std())
        print("mu   :", mu)
        print("sigma:", sigma)


# ===== main =====
if __name__ == "__main__":
    # ds = build_dataset()
    
    ds = normalize_images_only(ds) 
    verify(ds)

    print("\nIf you want to export:")
    print(">>> from astroclip_pipeline_subset import export_dataset")
    print(">>> export_dataset()")