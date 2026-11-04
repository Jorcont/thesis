"""
Convert raw JSON files (measurements + NWP) to Parquet.
Run once after 00_decompress.sh.

Outputs:
  trafo_meas_parquet/<asset_id>.parquet   – one Series: active_power_MW
  trafo_nwp_parquet/<tile_id>.parquet     – wide DataFrame: all NWP variables
  trafo_nwp_parquet/grid_coords.parquet   – tile lat/lon index
"""

import os
import json
import time
import pandas as pd

BASE_DIR     = "/mnt/data/kes/data/jdomingo/trafo_raw_data"
MEAS_DIR     = f"{BASE_DIR}/measurements_unpacked/meas_json_fomat"   # ← FIX 1: inner dir
NWP_DIR      = f"{BASE_DIR}/NWP_unpacked"
MEAS_PARQUET = "/mnt/data/kes/data/jdomingo/trafo_meas_parquet"
NWP_PARQUET  = "/mnt/data/kes/data/jdomingo/trafo_nwp_parquet"

os.makedirs(MEAS_PARQUET, exist_ok=True)
os.makedirs(NWP_PARQUET,  exist_ok=True)

NWP_VARIABLES = [
    "DIFFUSE_HORIZONTAL_IRRADIANCE",
    "DIRECT_HORIZONTAL_IRRADIANCE",
    "HUMIDITY_RELATIVE",
    "PRESSURE_SURFACE",
    "TEMPERATURE_2METER_ABOVE_GROUND",
    "WIND_DEGREES_40M",
    "WIND_DEGREES_100M",
    "WIND_SPEED_40M",
    "WIND_SPEED_100M",
]


def parse_json_to_series(path: str, value_key: str) -> pd.Series:
    """Load a JSON file and return a UTC-indexed pd.Series of values."""
    with open(path) as f:
        data = json.load(f)
    records = data[value_key]
    df = pd.DataFrame(records)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("t")["v"]


# ── 1. Measurements ───────────────────────────────────────────
print("Converting measurements...", flush=True)
meas_files = sorted(f for f in os.listdir(MEAS_DIR) if f.endswith(".json"))
print(f"  Found {len(meas_files)} measurement JSON files")

for fname in meas_files:
    asset_id = fname.replace(".json", "")
    out_path = f"{MEAS_PARQUET}/{asset_id}.parquet"
    if os.path.exists(out_path):
        continue
    try:
        s = parse_json_to_series(f"{MEAS_DIR}/{fname}", value_key="values")
        s.name = "active_power_MW"
        s.to_frame().to_parquet(out_path)
    except Exception as e:
        print(f"  ❌ {asset_id}: {e}")

print("  ✔ Measurements done\n", flush=True)

# ── 2. NWP grid coords ────────────────────────────────────────
grid_csv_candidates = [
    f"{BASE_DIR}/icon_eu_grid_export.csv",
    f"{BASE_DIR}/NWP_unpacked/icon_eu_grid_export.csv",
    f"{BASE_DIR}/NWP/icon_eu_grid_export.csv",
]
grid_csv = next((p for p in grid_csv_candidates if os.path.exists(p)), None)

if grid_csv is None:
    raise FileNotFoundError(
        "icon_eu_grid_export.csv not found. "
        "This file is required for NWP tile mapping. "
        "Check if it is inside one of the NWP zip files."
    )

grid_df = pd.read_csv(grid_csv, sep=";")
grid_df.columns = [c.strip().lower() for c in grid_df.columns]

id_col = next(c for c in grid_df.columns if "id" in c or "asset" in c)
grid_df = grid_df.rename(columns={"ids": "asset_id"}).set_index("asset_id")[["lats", "lons"]]
grid_df.to_parquet(f"{NWP_PARQUET}/grid_coords.parquet")
print(f"Saved grid_coords.parquet ({len(grid_df)} tiles)\n", flush=True)

# ── 3. NWP → wide Parquet (one file per tile, all variables as columns) ──
first_var      = NWP_VARIABLES[0]
first_var_dir  = f"{NWP_DIR}/{first_var}/{first_var}"                  # ← FIX 2: double dir
tile_ids = sorted(f.replace(".json", "") for f in os.listdir(first_var_dir)
                  if f.endswith(".json"))
print(f"Converting NWP: {len(tile_ids)} tiles × {len(NWP_VARIABLES)} variables",
      flush=True)

t0     = time.time()
failed = []

for i, tile_id in enumerate(tile_ids):
    out_path = f"{NWP_PARQUET}/{tile_id}.parquet"
    if os.path.exists(out_path):
        continue

    series_list = []
    ok = True

    for var in NWP_VARIABLES:
        fpath = f"{NWP_DIR}/{var}/{var}/{tile_id}.json"                # ← FIX 2: double dir
        if not os.path.exists(fpath):
            print(f"  ⚠ missing {var} for {tile_id}")
            ok = False
            break
        try:
            s = parse_json_to_series(fpath, value_key="expectationValues")
            s.name = var
            series_list.append(s)
        except Exception as e:
            print(f"  ❌ {tile_id}/{var}: {e}")
            ok = False
            break

    if not ok:
        failed.append(tile_id)
        continue

    df = pd.concat(series_list, axis=1)
    df.index.name = "t"
    df.to_parquet(out_path)

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        eta     = (len(tile_ids) - i - 1) / rate / 60
        print(f"  ✔ {i+1}/{len(tile_ids)} | {elapsed:.0f}s | ETA {eta:.1f} min",
              flush=True)

print(f"\n🏁 NWP done in {(time.time()-t0)/60:.1f} min | failed: {len(failed)}",
      flush=True)
if failed:
    print(f"   {failed}")
