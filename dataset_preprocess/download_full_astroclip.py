from datasets import load_dataset
from pathlib import Path

base_dir = Path("/scratch/users/nus/e1553819/astroclip")
cache_dir = base_dir / "hf_cache"

base_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

print("Base dir:", base_dir, flush=True)
print("Cache dir:", cache_dir, flush=True)
print("Start downloading full AstroCLIP dataset...", flush=True)

dset = load_dataset(
    "mhsotoudeh/astroclip",
    cache_dir=str(cache_dir)
)

print("\nDownload complete!", flush=True)
print("Train size:", len(dset["train"]), flush=True)
print("Validation size:", len(dset["validation"]), flush=True)
print("Test size:", len(dset["test"]), flush=True)
print("Sample keys:", dset["train"][0].keys(), flush=True)
