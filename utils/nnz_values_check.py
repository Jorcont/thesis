import os
import glob
import pandas as pd

PLANTS_DIR = "/mnt/data/kes/data/jdomingo/plants_clean"
TARGET_COL = "active_power_kW"
ZERO_TOL   = 0.005            # use e.g. 0.005 to count "near-zero" (night)

rows = []
files = sorted(glob.glob(os.path.join(PLANTS_DIR, "*.parquet")))

for f in files:
    plant_id = os.path.basename(f).replace(".parquet", "")
    try:
        s = pd.read_parquet(f, columns=[TARGET_COL])[TARGET_COL]
    except Exception as e:
        print(f"❌ {plant_id}: {e}")
        continue

    n_total = len(s)
    n_nan   = int(s.isna().sum())
    n_zero  = int((s.abs() <= ZERO_TOL).sum())      # exact zeros (or ≤ tol)
    pct_zero = 100 * n_zero / n_total if n_total else 0.0

    rows.append({
        "plant_id":   plant_id,
        "n_total":    n_total,
        "n_nan":      n_nan,
        "n_zero":     n_zero,
        "pct_zero":   round(pct_zero, 2),
    })

df = pd.DataFrame(rows).sort_values("pct_zero", ascending=False)

# ── Per-plant table ───────────────────────────────────────────
# print(df.to_string(index=False))

# ── Overall statistics ────────────────────────────────────────
print("\n=== Overall ===")
print(df[["pct_zero", "n_zero", "n_total"]].describe().round(2).to_string())
print(f"\nGlobal zero share: "
      f"{100 * df['n_zero'].sum() / df['n_total'].sum():.2f}%")

# optional: save
df.to_csv("zero_target_stats.csv", index=False)


----

import os
import glob
import numpy as np
import pandas as pd

PLANTS_DIR   = "/mnt/data/kes/data/jdomingo/plants_clean"
TARGET_COL   = "active_power_kW"
THRESHOLD    = 0.005      # non-zero cutoff
STEPS_PER_DAY = 96        # 4 * 24  (15-min data)
MIN_NNZ_PER_DAY = 10       # a day is "valid" if it has >= this many non-zero steps
SEQ_LENS     = [3, 4]     # consecutive-day windows to evaluate

rows = []
files = sorted(glob.glob(os.path.join(PLANTS_DIR, "*.parquet")))

for f in files:
    plant_id = os.path.basename(f).replace(".parquet", "")
    try:
        s = pd.read_parquet(f, columns=[TARGET_COL])[TARGET_COL]
    except Exception as e:
        print(f"❌ {plant_id}: {e}")
        continue

    vals = s.to_numpy()
    # trim to whole days
    n_days = len(vals) // STEPS_PER_DAY
    if n_days == 0:
        continue
    day_mat = vals[:n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)

    # non-zero count per day (NaN treated as zero/invalid)
    nnz_per_day = np.nansum(np.abs(day_mat) > THRESHOLD, axis=1)
    day_valid   = nnz_per_day >= MIN_NNZ_PER_DAY          # bool per day

    n_valid_days = int(day_valid.sum())
    pct_valid_days = 100 * n_valid_days / n_days

    rec = {
        "plant_id":       plant_id,
        "n_days":         n_days,
        "n_valid_days":   n_valid_days,
        "pct_valid_days": round(pct_valid_days, 2),
    }

    # ── consecutive-day sequences: every day in the window must be valid ──
    for L in SEQ_LENS:
        if n_days >= L:
            # sliding window over day_valid, all True?
            windows = np.lib.stride_tricks.sliding_window_view(day_valid, L)
            n_total_seq = windows.shape[0]
            n_valid_seq = int(windows.all(axis=1).sum())
            pct_seq = 100 * n_valid_seq / n_total_seq
        else:
            n_total_seq = n_valid_seq = 0
            pct_seq = 0.0
        rec[f"n_valid_{L}day"]   = n_valid_seq
        rec[f"pct_valid_{L}day"] = round(pct_seq, 2)

    rows.append(rec)

df = pd.DataFrame(rows).sort_values("pct_valid_days", ascending=False)

# ── Per-plant table ───────────────────────────────────────────
print(df.to_string(index=False))

# ── Overall summary ───────────────────────────────────────────
print("\n=== Overall (mean across plants) ===")
summary_cols = ["pct_valid_days"] + [f"pct_valid_{L}day" for L in SEQ_LENS]
print(df[summary_cols].describe().round(2).to_string())

df.to_csv("window_validity_stats.csv", index=False)

----


import os
import glob
import numpy as np
import pandas as pd

PLANTS_DIR      = "/mnt/data/kes/data/jdomingo/plants_clean"
TARGET_COL      = "active_power_kW"
THRESHOLD       = 0.005      # non-zero cutoff

STEPS_PER_DAY   = 96
LOOK_BACK       = 288        # steps
LOOK_AHEAD      = 96         # steps
WINDOW_STEPS    = LOOK_BACK + LOOK_AHEAD   # = 384; set to 288 if you prefer
STRIDE          = 1          # sliding step; use 96 for non-overlapping days

MIN_NNZ_PER_DAY = 10         # each 96-step day inside window must have >= this
REQUIRE_ALL_DAYS = True      # True: every day valid; False: whole-window nnz only
MIN_NNZ_WINDOW  = 1          # used only when REQUIRE_ALL_DAYS = False

rows = []
files = sorted(glob.glob(os.path.join(PLANTS_DIR, "*.parquet")))

for f in files:
    plant_id = os.path.basename(f).replace(".parquet", "")
    try:
        s = pd.read_parquet(f, columns=[TARGET_COL])[TARGET_COL]
    except Exception as e:
        print(f"❌ {plant_id}: {e}")
        continue

    vals = s.to_numpy()
    n = len(vals)
    if n < WINDOW_STEPS:
        continue

    # non-zero mask over the full series (NaN -> not counted)
    nnz_mask = (np.abs(vals) > THRESHOLD)
    nnz_mask = np.where(np.isnan(vals), False, nnz_mask)

    # sliding windows over the mask
    win = np.lib.stride_tricks.sliding_window_view(nnz_mask, WINDOW_STEPS)[::STRIDE]
    n_windows = win.shape[0]

    if REQUIRE_ALL_DAYS:
        # only works cleanly when WINDOW_STEPS is a multiple of STEPS_PER_DAY
        n_full_days = WINDOW_STEPS // STEPS_PER_DAY
        usable      = n_full_days * STEPS_PER_DAY
        # reshape each window into (n_windows, n_full_days, STEPS_PER_DAY)
        day_blocks  = win[:, :usable].reshape(n_windows, n_full_days, STEPS_PER_DAY)
        nnz_per_day = day_blocks.sum(axis=2)                    # (n_windows, n_days)
        valid       = (nnz_per_day >= MIN_NNZ_PER_DAY).all(axis=1)
    else:
        valid = win.sum(axis=1) >= MIN_NNZ_WINDOW

    n_valid = int(valid.sum())
    pct     = 100 * n_valid / n_windows if n_windows else 0.0

    rows.append({
        "plant_id":      plant_id,
        "n_windows":     n_windows,
        "n_valid":       n_valid,
        "pct_valid":     round(pct, 2),
    })

df = pd.DataFrame(rows).sort_values("pct_valid", ascending=False)

# ── Per-plant table ───────────────────────────────────────────
print(df.to_string(index=False))

# ── Overall summary ───────────────────────────────────────────
print(f"\n=== Overall (window={WINDOW_STEPS} steps, "
      f"min_nnz/day={MIN_NNZ_PER_DAY}, thr={THRESHOLD}) ===")
print(df[["pct_valid", "n_valid", "n_windows"]].describe().round(2).to_string())
print(f"\nGlobal valid share: "
      f"{100 * df['n_valid'].sum() / df['n_windows'].sum():.2f}%")
print(f"Total valid windows across all plants: {df['n_valid'].sum():,}")

df.to_csv("window_validity_true.csv", index=False)