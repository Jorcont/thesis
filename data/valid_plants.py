import os, glob
import numpy as np
import pandas as pd

PLANTS_DIR      = "data/PV_data/"  # NaN done
TARGET_COL      = "active_power_kW"
THRESHOLD       = 0.002
STEPS_PER_DAY   = 96
LOOK_BACK       = 288
LOOK_AHEAD      = 96
WINDOW_STEPS    = LOOK_BACK + LOOK_AHEAD    # 384
STRIDE          = 4
MIN_NNZ_PER_DAY = 18                        # Germany-appropriate
PCT_KEEP        = 75.0                       # keep plants >= this
OUT_TXT         = os.path.join(PLANTS_DIR, "valid_plants2.txt")
OUT_CSV         = os.path.join(PLANTS_DIR, "window_validity_final.csv")

rows = []
for f in sorted(glob.glob(os.path.join(PLANTS_DIR, "*.parquet"))):
    pid = os.path.basename(f).replace(".parquet", "")
    try:
        vals = pd.read_parquet(f, columns=[TARGET_COL])[TARGET_COL].to_numpy()
    except Exception as e:
        print(f"❌ {pid}: {e}");  continue
    if len(vals) < WINDOW_STEPS:
        continue

    nnz = np.where(np.isnan(vals), False, np.abs(vals) > THRESHOLD)
    win = np.lib.stride_tricks.sliding_window_view(nnz, WINDOW_STEPS)[::STRIDE]
    n_windows = win.shape[0]

    n_days = WINDOW_STEPS // STEPS_PER_DAY
    blocks = win[:, :n_days*STEPS_PER_DAY].reshape(n_windows, n_days, STEPS_PER_DAY)
    valid  = (blocks.sum(axis=2) >= MIN_NNZ_PER_DAY).all(axis=1)

    n_valid = int(valid.sum())
    rows.append({
        "plant_id": pid,
        "n_windows": n_windows,
        "n_valid": n_valid,
        "pct_valid": round(100 * n_valid / n_windows, 2) if n_windows else 0.0,
    })

df = pd.DataFrame(rows).sort_values("pct_valid", ascending=False)
df.to_csv(OUT_CSV, index=False)

keep = df.loc[df["pct_valid"] >= PCT_KEEP, "plant_id"].tolist()
with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(keep))

print(f"Total plants   : {len(df)}")
print(f"Kept (>= {PCT_KEEP}%): {len(keep)}")
print(f"Dropped        : {len(df) - len(keep)}")
print(f"→ {OUT_TXT}")
