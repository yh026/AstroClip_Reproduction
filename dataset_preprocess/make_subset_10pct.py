from datasets import load_dataset, concatenate_datasets
from pathlib import Path

CACHE_DIR = "/scratch/users/nus/e1553819/astroclip/hf_cache"
SAVE_DIR = "/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10"

SEED = 42
SUBSET_FRACTION = 0.10   # 10%

print("Loading full dataset from cache...")
dset = load_dataset(
    "mhsotoudeh/astroclip",
    cache_dir=CACHE_DIR
)

print("Merging train/val/test into one full dataset...")
full_dataset = concatenate_datasets([
    dset["train"],
    dset["val"],
    dset["test"]
])

total_size = len(full_dataset)
subset_size = int(total_size * SUBSET_FRACTION)

print("Full dataset size:", total_size)
print("Subset size (10%):", subset_size)

# 1) shuffle once for reproducibility
full_dataset = full_dataset.shuffle(seed=SEED)

# 2) take first 10%
subset_10pct = full_dataset.select(range(subset_size))

print("10% subset created:", len(subset_10pct))

# 3) split subset into 90/10 train/test
split = subset_10pct.train_test_split(
    test_size=0.1,
    seed=SEED
)

print("Train size:", len(split["train"]))
print("Test size:", len(split["test"]))

# 4) save to disk
save_path = Path(SAVE_DIR)
save_path.parent.mkdir(parents=True, exist_ok=True)

print("Saving subset to disk...")
split.save_to_disk(str(save_path))

print("Saved successfully to:", save_path)
print(split)