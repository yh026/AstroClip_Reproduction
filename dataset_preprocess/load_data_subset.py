from datasets import load_from_disk

DATA_PATH = "/scratch/users/nus/e1553819/astroclip/shared_subset_10pct_90_10"

def load_astroclip_subset():
    dset = load_from_disk(DATA_PATH)
    return dset["train"], dset["test"]
