import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/mnt/data/kes/data/jdomingo/meta-netze")
sys.path.insert(0, "/mnt/data/kes/data/jdomingo/pyolorin-data-preprocessing")

from data.parquet_loader import load_single_nwp, load_plant_nwp_mapping
from pyolorin_preprocessor.build_features import BuildFeatures
from feature_scaler import FeatureScaler

GOOD_TARGET_DIR = "/mnt/data/kes/data/jdomingo/plants_clean"
OUT_DIR = "/mnt/data/kes/data/jdomingo/plants_clean_good"
os.makedirs(OUT_DIR, exist_ok=True)

BF_CONFIG = {
    "input_features": {
        "self_built": {
            "value_type": [
                "SOLAR_ELEVATION",
                "SOLAR_AZIMUTH",
                "TIME_OF_DAY_SIN",
                "TIME_OF_DAY_COS",
                "DAY_OF_YEAR_SIN",
                "DAY_OF_YEAR_COS",
            ]
        }
    }
}


def build_feature_df(nwp_tile, lat, lon):
    nwp = load_single_nwp(nwp_tile)
    nwp = nwp[~nwp.index.duplicated(keep="first")]

    full = pd.date_range(nwp.index.min(), nwp.index.max(), freq="15min", tz="UTC")
    nwp = nwp.reindex(full).interpolate(method="time")

    master = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    bf = BuildFeatures(live_inference_mode=False)

    df_features = bf.build(
        config=BF_CONFIG,
        input_df=nwp,
        master_data_asset=master
    ).reindex(full)

    return df_features


def accumulate_global_stats(df_features, acc):
    for col in FeatureScaler.ZSCORE_COLS:
        if col in df_features.columns:
            s = df_features[col].dropna()
            if len(s) == 0:
                continue

            if col not in acc["z"]:
                acc["z"][col] = {"n": 0, "sum": 0.0, "sumsq": 0.0}

            acc["z"][col]["n"] += int(len(s))
            acc["z"][col]["sum"] += float(s.sum())
            acc["z"][col]["sumsq"] += float((s ** 2).sum())

    for col in FeatureScaler.MINMAX_LB0:
        if col in df_features.columns:
            s = df_features[col].dropna()
            if len(s) == 0:
                continue
            mx = float(s.max())
            acc["max"][col] = max(acc["max"].get(col, -np.inf), mx)

    return acc


def finalize_global_stats(acc):
    zstats = {}

    for col, a in acc["z"].items():
        n = a["n"]
        mu = a["sum"] / n if n > 0 else 0.0

        if n > 1:
            var = (a["sumsq"] - (a["sum"] ** 2) / n) / (n - 1)
            var = max(var, 0.0)
            sd = float(np.sqrt(var))
        else:
            sd = 1.0

        zstats[col] = (float(mu), float(sd if sd > 1e-8 else 1.0))

    maxvals = {
        col: float(mx if mx > 1e-8 else 1.0)
        for col, mx in acc["max"].items()
    }

    return zstats, maxvals


def save_global_stats(zstats, maxvals):
    payload = {
        "zstats": {k: [float(v[0]), float(v[1])] for k, v in zstats.items()},
        "maxvals": {k: float(v) for k, v in maxvals.items()},
    }

    with open(f"{OUT_DIR}/global_stats.json", "w") as f:
        json.dump(payload, f, indent=2)


def run():
    mapping = load_plant_nwp_mapping()
    have = {
        f.replace(".parquet", "")
        for f in os.listdir(GOOD_TARGET_DIR)
        if f.endswith(".parquet")
    }

    mapping = mapping[mapping["asset_id"].isin(have)].reset_index(drop=True)
    print(f"Plants to process: {len(mapping)}", flush=True)

    # PASS 1: compute global stats
    print("Pass 1/2: computing global stats...", flush=True)
    acc = {"z": {}, "max": {}}

    for i, row in mapping.iterrows():
        try:
            df_features = build_feature_df(
                row["nwp_tile"],
                row["LATITUDE"],
                row["LONGITUDE"]
            )
            accumulate_global_stats(df_features, acc)
        except Exception as e:
            print(f"  ⚠ stats skip {row['asset_id']}: {e}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  stats {i+1}/{len(mapping)}", flush=True)

    zstats, maxvals = finalize_global_stats(acc)
    save_global_stats(zstats, maxvals)
    print("Saved global_stats.json", flush=True)

    # PASS 2: transform and write final parquet
    print("Pass 2/2: scaling and writing files...", flush=True)
    t0 = time.time()
    failed = []

    scaler = FeatureScaler().set_global_stats(zstats, maxvals)

    for i, row in mapping.iterrows():
        pid = row["asset_id"]
        dst = f"{OUT_DIR}/{pid}.parquet"

        if os.path.exists(dst):
            continue

        try:
            df_features = build_feature_df(
                row["nwp_tile"],
                row["LATITUDE"],
                row["LONGITUDE"]
            )

            scaled = scaler.transform(df_features)

            good = pd.read_parquet(
                f"{GOOD_TARGET_DIR}/{pid}.parquet",
                columns=["active_power_kW"]
            )

            df_out = scaled.join(good["active_power_kW"], how="inner")
            df_out.to_parquet(dst)

        except Exception as e:
            print(f"  ❌ {pid}: {e}", flush=True)
            failed.append(pid)
            continue

        if (i + 1) % 100 == 0:
            print(f"  ✔ {i+1}/{len(mapping)} | {time.time()-t0:.0f}s", flush=True)

    print(f"\n🏁 Done in {(time.time()-t0)/60:.1f} min | failed: {len(failed)}", flush=True)
    if failed:
        print(failed, flush=True)


if __name__ == "__main__":
    run()