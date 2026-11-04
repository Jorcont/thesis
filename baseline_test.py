"""
sanity_check_baseline.py
Standalone check for PVTimeSeriesDatasetV2 + make_baseline_dataset.
Run: python sanity_check_baseline.py
"""
import os
import glob
import torch
from collections import Counter
from torch.utils.data import DataLoader, RandomSampler

from data.dataset_classes.PVTimeSeriesDatasetV2 import PVTimeSeriesDatasetV2
from data.dataset_classes.BaselineDataset import make_baseline_dataset

# ──────────────────────────────────────────────────────────
# 1) Minimal config — mirrors the relevant YAML fields
# ──────────────────────────────────────────────────────────
PLANTS_DIR = "data/PV_data"

cfg_dict = {
    "look_back_steps": 288,
    "look_ahead_steps": 96,
    "window_stride": 12,
    "train_ratio": 0.5,
    "gap": 96,
    "theta_support": 0.5,
    "theta_query": 0.5,
    # "max_windows_train": 500,
    # "max_windows_val": 500,
    # "max_windows_test": 500,
    # "time_varying_known_reals": [  # presence of this key => use_tft = True
    #     "SOLAR_ELEVATION", "SOLAR_AZIMUTH_SIN", "SOLAR_AZIMUTH_COS",
    #     "TIME_OF_DAY_SIN", "TIME_OF_DAY_COS", "DAY_OF_YEAR_SIN", "DAY_OF_YEAR_COS",
    # ],
    "use_pf_tft": False,
}

# >>> ADJUST THIS <<< to match your real DataConfig constructor
from conf.data_configs import DataConfig
DataConfig(name="PVTimeSeries", config=cfg_dict)   # sets the singleton
# If your DataConfig needs different args, adjust the line above only.

# ──────────────────────────────────────────────────────────
# 2) Pick a handful of real plant ids from PLANTS_DIR
# ──────────────────────────────────────────────────────────
parquet_files = sorted(glob.glob(os.path.join(PLANTS_DIR, "*.parquet")))
plant_ids = [
    os.path.splitext(os.path.basename(f))[0]
    for f in parquet_files
    if "_renorm_report" not in f
][:50]  # just a few plants for a quick check

if not plant_ids:
    raise RuntimeError(f"No parquet plant files found in {PLANTS_DIR}")

print(f"[1] Using {len(plant_ids)} plants: {plant_ids}")

# ──────────────────────────────────────────────────────────
# 3) Preload Arrow cache (class-level, required before init_data)
# ──────────────────────────────────────────────────────────
PVTimeSeriesDatasetV2.preload(plant_ids, PLANTS_DIR)

# ──────────────────────────────────────────────────────────
# 4) Build the BASELINE dataset (full_task-based, flat index)
# ──────────────────────────────────────────────────────────
BaselinePV = make_baseline_dataset(PVTimeSeriesDatasetV2)
ds = BaselinePV(pool=plant_ids)
ds.init_data()  # BaselineDataset.init_data() calls full_task() for each plant

print(f"\n[2] len(dataset) = {len(ds)}  (should be sum of per-plant window counts)")
print(f"    index_map[:5] = {ds.index_map[:50]}  (task_idx, sample_idx) pairs")

# ──────────────────────────────────────────────────────────
# 5) Inspect one sample's shape
# ──────────────────────────────────────────────────────────
x0, y0 = ds[0]
if isinstance(x0, tuple):
    print(f"\n[3] Sample 0 (TFT mode): X_enc.shape={x0[0].shape}, "
          f"X_dec.shape={x0[1].shape}, y.shape={y0.shape}")
else:
    print(f"\n[3] Sample 0 (LSTM mode): X.shape={x0.shape}, y.shape={y0.shape}")

# ──────────────────────────────────────────────────────────
# 6) WITHOUT replacement — one epoch via shuffle=True
# ──────────────────────────────────────────────────────────
loader_no_repl = DataLoader(ds, batch_size=4, shuffle=True)
seen = []
for _, y in loader_no_repl:
    seen.extend(range(len(y)))  # dummy; real check below uses indices directly

sampler_no_repl = list(torch.utils.data.RandomSampler(ds, replacement=False))
dup_no_repl = len(sampler_no_repl) - len(set(sampler_no_repl))
print(f"\n[4] WITHOUT replacement: drew {len(sampler_no_repl)} indices, "
      f"{dup_no_repl} duplicates (should be 0)")

# ──────────────────────────────────────────────────────────
# 7) WITH replacement — explicit RandomSampler
# ──────────────────────────────────────────────────────────
n_draws = len(ds) * 2  # force duplicates to be likely
sampler_with_repl = list(RandomSampler(ds, replacement=True, num_samples=n_draws))
counts = Counter(sampler_with_repl)
dup_with_repl = sum(c - 1 for c in counts.values() if c > 1)
print(f"\n[5] WITH replacement: drew {n_draws} indices, "
      f"{len(counts)} distinct, {dup_with_repl} duplicate draws (should be > 0)")

print("\n✅ Sanity check complete.")
