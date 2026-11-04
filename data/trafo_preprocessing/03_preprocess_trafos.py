# """
# Full 2-pass preprocessing pipeline for trafo active power.

# Pass 1  – NaN/outlier report + accumulate global NWP feature stats
# Pass 2  – Scale features, fill NaN target, per-trafo min-max normalise, save

# Output per trafo: <OUT_DIR>/<asset_id>.parquet
#   Index : UTC DatetimeIndex, 15-min resolution (epoch ms in parquet)
#   Cols  : all scaled NWP + derived features + active_power_MW [0, 1]

# Side outputs:
#   nan_outlier_report.csv  – per-trafo NaN & outlier summary
#   global_stats.json       – z-score params, NWP max-vals, per-trafo target max
# """

# import os
# import sys
# import json
# import time
# import warnings
# import numpy as np
# import pandas as pd

# warnings.filterwarnings("ignore")

# sys.path.insert(0, "/mnt/data/kes/data/jdomingo/meta-netze")
# sys.path.insert(0, "/mnt/data/kes/data/jdomingo/pyolorin-data-preprocessing")

# from pyolorin_preprocessor.interpolation import Interpolation
# from pyolorin_preprocessor.build_features import BuildFeatures
# from pyolorin_preprocessor.fill_nan import FillNan          # reuse _enforce_max_gaps
# from trafo_feature_scaler import TrafoFeatureScaler

# # ── Paths ──────────────────────────────────────────────────────
# MEAS_PARQUET = "/mnt/data/kes/data/jdomingo/trafo_meas_parquet"
# NWP_PARQUET  = "/mnt/data/kes/data/jdomingo/trafo_nwp_parquet"
# OUT_DIR      = "/mnt/data/kes/data/jdomingo/trafo_preprocessed"
# os.makedirs(OUT_DIR, exist_ok=True)

# # ── Pipeline configs ───────────────────────────────────────────
# INTERP_CONFIG = {
#     "interpolation": {
#         "resample_freq": "15min",
#         "style": {
#             "default":                        "linear",
#             "DIRECT_HORIZONTAL_IRRADIANCE":   "solar",
#             "DIFFUSE_HORIZONTAL_IRRADIANCE":  "solar",
#         },
#     }
# }

# BF_CONFIG = {
#     "input_features": {
#         "self_built": {
#             "value_type": [
#                 "SOLAR_ELEVATION",
#                 "SOLAR_AZIMUTH",
#                 "TIME_OF_DAY_SIN",
#                 "TIME_OF_DAY_COS",
#                 "DAY_OF_YEAR_SIN",
#                 "DAY_OF_YEAR_COS",
#             ]
#         }
#     }
# }

# FILL_CONFIG = {
#     "fill_nan_training": {
#         "max_mid_gap":   "2h",
#         "max_lead_gap":  "2h",
#         "max_trail_gap": "2h",
#     }
# }

# IQR_MULTIPLIER = 3.0


# # ══════════════════════════════════════════════════════════════
# #  Helper functions
# # ══════════════════════════════════════════════════════════════

# def load_mapping() -> pd.DataFrame:
#     return pd.read_parquet(f"{NWP_PARQUET}/trafo_nwp_mapping.parquet")


# def load_measurement(asset_id: str) -> pd.Series:
#     df = pd.read_parquet(f"{MEAS_PARQUET}/{asset_id}.parquet")
#     s  = df["active_power_MW"]
#     # Deduplicate timestamps
#     return s[~s.index.duplicated(keep="first")]


# def load_nwp_raw(tile_id: str) -> pd.DataFrame:
#     df = pd.read_parquet(f"{NWP_PARQUET}/{tile_id}.parquet")
#     return df[~df.index.duplicated(keep="first")]


# # ── NaN / Outlier diagnostics ─────────────────────────────────

# def nan_outlier_report(series: pd.Series, asset_id: str) -> dict:
#     """Return a dict summarising NaN count and IQR-based outliers."""
#     n_total = len(series)
#     n_nan   = int(series.isna().sum())
#     valid   = series.dropna()

#     q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
#     iqr     = q3 - q1
#     # lo      = max(0.0, q1 - IQR_MULTIPLIER * iqr)
#     lo      = q1 - IQR_MULTIPLIER * iqr

#     hi      = q3 + IQR_MULTIPLIER * iqr
#     n_out   = int(((valid < lo) | (valid > hi)).sum())

#     return {
#         "asset_id":       asset_id,
#         "n_total":        n_total,
#         "n_nan":          n_nan,
#         "pct_nan":        round(100 * n_nan / n_total, 2),
#         "n_outliers_iqr": n_out,
#         "pct_outliers":   round(100 * n_out / max(len(valid), 1), 2),
#         "min_raw":        round(float(valid.min()),  4) if len(valid) else None,
#         "max_raw":        round(float(valid.max()),  4) if len(valid) else None,
#         "mean_raw":       round(float(valid.mean()), 4) if len(valid) else None,
#         "iqr_clip_lo":    round(lo, 4),
#         "iqr_clip_hi":    round(hi, 4),
#     }


# def clip_outliers(series: pd.Series) -> pd.Series:
#     """IQR-based clip; lower bound is always ≥ 0 (active power ≥ 0)."""
#     valid   = series.dropna()
#     q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
#     iqr     = q3 - q1
#     # lo      = max(0.0, q1 - IQR_MULTIPLIER * iqr)
#     lo      = q1 - IQR_MULTIPLIER * iqr
#     hi      = q3 + IQR_MULTIPLIER * iqr
#     return series.clip(lower=lo, upper=hi)


# # ── Target NaN filling (linear, not solar-based) ──────────────

# def fill_nan_trafo(series: pd.Series, config: dict) -> pd.Series:
#     """
#     Linear interpolation with max-gap enforcement.
#     Reuses FillNan._enforce_max_gaps (static, no solar dependency).
#     """
#     full_index = pd.date_range(
#         series.index.min(), series.index.max(), freq="15min", tz="UTC"
#     )
#     series = series.reindex(full_index)

#     cfg  = config.get("fill_nan_training", {})
#     filled_all = series.interpolate(method="time")

#     return FillNan._enforce_max_gaps(
#         pv_filled_all_times = filled_all,
#         pv_original         = series,
#         time_resolution     = pd.Timedelta("15min"),
#         max_mid_gap         = pd.to_timedelta(cfg.get("max_mid_gap",   "2h")),
#         max_lead_gap        = pd.to_timedelta(cfg.get("max_lead_gap",  "2h")),
#         max_trail_gap       = pd.to_timedelta(cfg.get("max_trail_gap", "2h")),
#     )


# # ── NWP feature building ──────────────────────────────────────

# def build_nwp_features(tile_id: str, lat: float, lon: float) -> pd.DataFrame:
#     """
#     Load raw hourly NWP → interpolate to 15 min (solar-aware for irradiance)
#     → append solar position + calendar features.
#     """
#     nwp_raw = load_nwp_raw(tile_id)
#     master  = pd.DataFrame({"latitude": [lat], "longitude": [lon]})

#     # Interpolation from pyolorin_preprocessor (do not modify)
#     df_15 = Interpolation.interpolate(
#         config            = INTERP_CONFIG,
#         input_df          = nwp_raw,
#         master_data_asset = master,
#     )

#     # Add solar elevation/azimuth + calendar sin/cos
#     bf     = BuildFeatures(live_inference_mode=False)
#     df_out = bf.build(
#         config            = BF_CONFIG,
#         input_df          = df_15,
#         master_data_asset = master,
#     )
#     return df_out


# # ── Global stats accumulation ─────────────────────────────────

# def accumulate_stats(df: pd.DataFrame, acc: dict) -> None:
#     """Update running accumulators for z-score and min-max stats."""
#     for col in TrafoFeatureScaler.ZSCORE_COLS:
#         if col not in df.columns:
#             continue
#         s = df[col].dropna()
#         if len(s) == 0:
#             continue
#         if col not in acc["z"]:
#             acc["z"][col] = {"n": 0, "sum": 0.0, "sumsq": 0.0}
#         acc["z"][col]["n"]     += int(len(s))
#         acc["z"][col]["sum"]   += float(s.sum())
#         acc["z"][col]["sumsq"] += float((s ** 2).sum())

#     for col in TrafoFeatureScaler.MINMAX_LB0:
#         if col not in df.columns:
#             continue
#         mx = float(df[col].dropna().max())
#         acc["max"][col] = max(acc["max"].get(col, -np.inf), mx)


# def finalize_stats(acc: dict) -> tuple[dict, dict]:
#     zstats = {}
#     for col, a in acc["z"].items():
#         n   = a["n"]
#         mu  = a["sum"] / n if n > 0 else 0.0
#         var = max((a["sumsq"] - a["sum"] ** 2 / n) / (n - 1), 0.0) if n > 1 else 1.0
#         sd  = float(np.sqrt(var))
#         zstats[col] = (float(mu), float(sd if sd > 1e-8 else 1.0))

#     maxvals = {c: float(max(v, 1e-8)) for c, v in acc["max"].items()}
#     return zstats, maxvals


# def save_global_stats(zstats: dict, maxvals: dict,
#                       target_max: dict) -> None:
#     payload = {
#         "feature_zstats":  {k: list(v)     for k, v in zstats.items()},
#         "feature_maxvals": maxvals,
#         "target_max_MW":   target_max,      # {asset_id: max_active_power_MW}
#     }
#     path = f"{OUT_DIR}/global_stats.json"
#     with open(path, "w") as f:
#         json.dump(payload, f, indent=2)
#     print(f"✅ Saved → {path}")


# # ══════════════════════════════════════════════════════════════
# #  Main pipeline
# # ══════════════════════════════════════════════════════════════

# def run():
#     mapping = load_mapping()

#     # Restrict to trafos that have measurement parquets
#     available = {
#         f.replace(".parquet", "")
#         for f in os.listdir(MEAS_PARQUET) if f.endswith(".parquet")
#     }
#     mapping = mapping[mapping["asset_id"].isin(available)].reset_index(drop=True)
#     print(f"Trafos to process: {len(mapping)}\n", flush=True)

#     # ──────────────────────────────────────────────────────────
#     # PASS 1  NaN/outlier diagnostics + global feature stats
#     # ──────────────────────────────────────────────────────────
#     print("Pass 1/2: diagnostics + global stats...", flush=True)
#     acc        = {"z": {}, "max": {}}
#     reports    = []
#     target_max = {}          # asset_id → max_MW after clipping
#     t0         = time.time()

#     for i, row in mapping.iterrows():
#         asset_id = row["asset_id"]
#         try:
#             series  = load_measurement(asset_id)
#             reports.append(nan_outlier_report(series, asset_id))

#             clipped = clip_outliers(series.dropna())
#             target_max[asset_id] = float(clipped.max()) if len(clipped) > 0 else 1.0

#             df_feat = build_nwp_features(
#                 row["nwp_tile"], row["latitude"], row["longitude"]
#             )
#             accumulate_stats(df_feat, acc)

#         except Exception as e:
#             print(f"  ⚠ pass1 skip {asset_id}: {e}", flush=True)

#         if (i + 1) % 10 == 0:
#             print(f"  stats {i+1}/{len(mapping)} | {time.time()-t0:.0f}s",
#                   flush=True)

#     # Save NaN/outlier report
#     report_df = pd.DataFrame(reports)
#     report_csv = f"{OUT_DIR}/nan_outlier_report.csv"
#     report_df.to_csv(report_csv, index=False)
#     print(f"\n--- NaN / Outlier Report (all trafos) ---")
#     print(report_df[["asset_id", "n_nan", "pct_nan",
#                      "n_outliers_iqr", "pct_outliers",
#                      "min_raw", "max_raw"]].to_string(index=False))
#     print(f"\nFull report saved → {report_csv}\n")

#     zstats, maxvals = finalize_stats(acc)
#     save_global_stats(zstats, maxvals, target_max)

#     # ──────────────────────────────────────────────────────────
#     # PASS 2  Scale features + fill NaN + normalise + save
#     # ──────────────────────────────────────────────────────────
#     print("\nPass 2/2: scaling + saving...", flush=True)
#     scaler = TrafoFeatureScaler().set_global_stats(zstats, maxvals)
#     t0     = time.time()
#     failed = []

#     for i, row in mapping.iterrows():
#         asset_id = row["asset_id"]
#         out_path = f"{OUT_DIR}/{asset_id}.parquet"

#         if os.path.exists(out_path):
#             continue

#         try:
#             # 1. Target: clip outliers → fill NaN (linear, gap-limited)
#             series = load_measurement(asset_id)
#             series = clip_outliers(series)
#             series = fill_nan_trafo(series, FILL_CONFIG)

#             # 2. NWP features: interpolate to 15 min + solar/calendar
#             df_feat = build_nwp_features(
#                 row["nwp_tile"], row["latitude"], row["longitude"]
#             )

#             # 3. Scale features globally
#             df_scaled = scaler.transform(df_feat)

#             # 4. Align target to the feature index (inner join on 15-min grid)
#             df_out = df_scaled.copy()
#             df_out["active_power_MW"] = series.reindex(df_scaled.index)

#             # 5. Per-trafo min-max normalisation of target
#             #    min = 0 (guaranteed ≥ 0), max = clipped max from pass 1
#             max_mw = max(target_max.get(asset_id, 1.0), 1e-8)
#             df_out["active_power_MW"] = (
#                 df_out["active_power_MW"] / max_mw
#             ).clip(lower=0.0)

#             # 6. Save — index = UTC DatetimeIndex (epoch ms in parquet)
#             df_out.index.name = "t"
#             df_out.to_parquet(out_path)

#         except Exception as e:
#             print(f"  ❌ {asset_id}: {e}", flush=True)
#             failed.append(asset_id)
#             continue

#         if (i + 1) % 10 == 0:
#             elapsed = time.time() - t0
#             rate    = max((i + 1) / elapsed, 1e-9)
#             eta     = (len(mapping) - i - 1) / rate / 60
#             print(f"  ✔ {i+1}/{len(mapping)} | "
#                   f"{elapsed:.0f}s | ETA {eta:.1f} min", flush=True)

#     print(f"\n🏁 Done in {(time.time()-t0)/60:.1f} min | "
#           f"failed: {len(failed)}", flush=True)
#     if failed:
#         print(f"   {failed}")


# if __name__ == "__main__":
#     run()

"""
Full 2-pass preprocessing pipeline for trafo active power.

Pass 1  – NaN/outlier report + accumulate global NWP feature stats
Pass 2  – Scale features, fill NaN target, per-trafo min-max normalise, save

Output per trafo: <OUT_DIR>/<asset_id>.parquet
  Index : UTC DatetimeIndex, 15-min resolution (epoch ms in parquet)
  Cols  : all scaled NWP + derived features + active_power_MW [0, 1]

Side outputs:
  nan_outlier_report.csv  – per-trafo NaN & outlier summary
  global_stats.json       – z-score params, NWP max-vals, per-trafo target stats
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/mnt/data/kes/data/jdomingo/meta-netze")
sys.path.insert(0, "/mnt/data/kes/data/jdomingo/pyolorin-data-preprocessing")

from pyolorin_preprocessor.interpolation import Interpolation
from pyolorin_preprocessor.build_features import BuildFeatures
from pyolorin_preprocessor.fill_nan import FillNan
from trafo_feature_scaler import TrafoFeatureScaler

# ── Paths ──────────────────────────────────────────────────────
MEAS_PARQUET = "/mnt/data/kes/data/jdomingo/trafo_meas_parquet"
NWP_PARQUET  = "/mnt/data/kes/data/jdomingo/trafo_nwp_parquet"
OUT_DIR      = "/mnt/data/kes/data/jdomingo/trafo_preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Train/test split date ──────────────────────────────────────
SPLIT_DATE        = "2024-01-01"   # adjust to your actual split
MIN_TEST_COVERAGE = 0.50           # exclude if test period has < 50% valid data

# ── Pipeline configs ───────────────────────────────────────────
INTERP_CONFIG = {
    "interpolation": {
        "resample_freq": "15min",
        "style": {
            "default":                        "linear",
            "DIRECT_HORIZONTAL_IRRADIANCE":   "solar",
            "DIFFUSE_HORIZONTAL_IRRADIANCE":  "solar",
        },
    }
}

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

FILL_CONFIG = {
    "fill_nan_training": {
        "max_mid_gap":   "2h",
        "max_lead_gap":  "2h",
        "max_trail_gap": "2h",
    }
}

IQR_MULTIPLIER = 3.0


# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def load_mapping() -> pd.DataFrame:
    return pd.read_parquet(f"{NWP_PARQUET}/trafo_nwp_mapping.parquet")


def load_measurement(asset_id: str) -> pd.Series:
    df = pd.read_parquet(f"{MEAS_PARQUET}/{asset_id}.parquet")
    s  = df["active_power_MW"]
    return s[~s.index.duplicated(keep="first")]


def load_nwp_raw(tile_id: str) -> pd.DataFrame:
    df = pd.read_parquet(f"{NWP_PARQUET}/{tile_id}.parquet")
    return df[~df.index.duplicated(keep="first")]


# ── NaN / Outlier diagnostics ─────────────────────────────────

def nan_outlier_report(series: pd.Series, asset_id: str) -> dict:
    """Return a dict summarising NaN count, IQR-based outliers, and train/test NaN split."""
    n_total = len(series)
    n_nan   = int(series.isna().sum())
    valid   = series.dropna()

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr     = q3 - q1
    lo      = q1 - IQR_MULTIPLIER * iqr
    hi      = q3 + IQR_MULTIPLIER * iqr
    n_out   = int(((valid < lo) | (valid > hi)).sum())

    # ── Train/test NaN split ──────────────────────────────────
    train_series = series[:SPLIT_DATE]
    test_series  = series[SPLIT_DATE:]
    train_nan_pct = round(100 * train_series.isna().mean(), 2) if len(train_series) > 0 else None
    test_nan_pct  = round(100 * test_series.isna().mean(),  2) if len(test_series)  > 0 else None
    # ─────────────────────────────────────────────────────────

    return {
        "asset_id":        asset_id,
        "n_total":         n_total,
        "n_nan":           n_nan,
        "pct_nan":         round(100 * n_nan / n_total, 2),
        "train_nan_pct":   train_nan_pct,
        "test_nan_pct":    test_nan_pct,
        "n_outliers_iqr":  n_out,
        "pct_outliers":    round(100 * n_out / max(len(valid), 1), 2),
        "min_raw":         round(float(valid.min()),  4) if len(valid) else None,
        "max_raw":         round(float(valid.max()),  4) if len(valid) else None,
        "mean_raw":        round(float(valid.mean()), 4) if len(valid) else None,
        "iqr_clip_lo":     round(lo, 4),
        "iqr_clip_hi":     round(hi, 4),
    }


def clip_outliers(series: pd.Series) -> pd.Series:
    """IQR-based clip over full bidirectional range (no forced lower bound)."""
    valid   = series.dropna()
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr     = q3 - q1
    lo      = q1 - IQR_MULTIPLIER * iqr
    hi      = q3 + IQR_MULTIPLIER * iqr
    return series.clip(lower=lo, upper=hi)


def is_degenerate(series: pd.Series) -> bool:
    """True if signal is flat (min == max) → normalization undefined."""
    valid = series.dropna()
    return len(valid) == 0 or float(valid.max() - valid.min()) < 1e-8


def exclude_by_test_nan(series: pd.Series) -> bool:
    """True if test period has insufficient valid data (< MIN_TEST_COVERAGE)."""
    test_series = series[SPLIT_DATE:]
    if len(test_series) == 0:
        return True
    valid_frac = 1.0 - test_series.isna().mean()
    return valid_frac < MIN_TEST_COVERAGE


# ── Target NaN filling (linear, not solar-based) ──────────────

def fill_nan_trafo(series: pd.Series, config: dict) -> pd.Series:
    """Linear interpolation with max-gap enforcement."""
    full_index = pd.date_range(
        series.index.min(), series.index.max(), freq="15min", tz="UTC"
    )
    series = series.reindex(full_index)

    cfg        = config.get("fill_nan_training", {})
    filled_all = series.interpolate(method="time")

    return FillNan._enforce_max_gaps(
        pv_filled_all_times = filled_all,
        pv_original         = series,
        time_resolution     = pd.Timedelta("15min"),
        max_mid_gap         = pd.to_timedelta(cfg.get("max_mid_gap",   "2h")),
        max_lead_gap        = pd.to_timedelta(cfg.get("max_lead_gap",  "2h")),
        max_trail_gap       = pd.to_timedelta(cfg.get("max_trail_gap", "2h")),
    )


# ── NWP feature building ──────────────────────────────────────

def build_nwp_features(tile_id: str, lat: float, lon: float) -> pd.DataFrame:
    nwp_raw = load_nwp_raw(tile_id)
    master  = pd.DataFrame({"latitude": [lat], "longitude": [lon]})

    df_15  = Interpolation.interpolate(
        config=INTERP_CONFIG, input_df=nwp_raw, master_data_asset=master,
    )
    bf     = BuildFeatures(live_inference_mode=False)
    df_out = bf.build(
        config=BF_CONFIG, input_df=df_15, master_data_asset=master,
    )

    # Circular encoding for angular features
    for deg_col in ["WIND_DEGREES_40M", "WIND_DEGREES_100M", "SOLAR_AZIMUTH"]:
        if deg_col in df_out.columns:
            rad = np.radians(df_out[deg_col])
            df_out[f"{deg_col}_SIN"] = np.sin(rad)
            df_out[f"{deg_col}_COS"] = np.cos(rad)
            df_out.drop(columns=[deg_col], inplace=True)

    return df_out


# ── Global stats accumulation ─────────────────────────────────

def accumulate_stats(df: pd.DataFrame, acc: dict) -> None:
    for col in TrafoFeatureScaler.ZSCORE_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if col not in acc["z"]:
            acc["z"][col] = {"n": 0, "sum": 0.0, "sumsq": 0.0}
        acc["z"][col]["n"]     += int(len(s))
        acc["z"][col]["sum"]   += float(s.sum())
        acc["z"][col]["sumsq"] += float((s ** 2).sum())

    for col in TrafoFeatureScaler.MINMAX_LB0:
        if col not in df.columns:
            continue
        mx = float(df[col].dropna().max())
        acc["max"][col] = max(acc["max"].get(col, -np.inf), mx)


def finalize_stats(acc: dict) -> tuple[dict, dict]:
    zstats = {}
    for col, a in acc["z"].items():
        n   = a["n"]
        mu  = a["sum"] / n if n > 0 else 0.0
        var = max((a["sumsq"] - a["sum"] ** 2 / n) / (n - 1), 0.0) if n > 1 else 1.0
        sd  = float(np.sqrt(var))
        zstats[col] = (float(mu), float(sd if sd > 1e-8 else 1.0))

    maxvals = {c: float(max(v, 1e-8)) for c, v in acc["max"].items()}
    return zstats, maxvals


def save_global_stats(zstats: dict, maxvals: dict, target_stats: dict) -> None:
    payload = {
        "feature_zstats":  {k: list(v) for k, v in zstats.items()},
        "feature_maxvals": maxvals,
        "target_stats":    target_stats,   # {asset_id: {"min": ..., "max": ...}}
    }
    path = f"{OUT_DIR}/global_stats.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {path}")


# ══════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════

def run():
    mapping = load_mapping()

    available = {
        f.replace(".parquet", "")
        for f in os.listdir(MEAS_PARQUET) if f.endswith(".parquet")
    }
    mapping = mapping[mapping["asset_id"].isin(available)].reset_index(drop=True)
    print(f"Trafos to process: {len(mapping)}\n", flush=True)

    # ──────────────────────────────────────────────────────────
    # PASS 1  NaN/outlier diagnostics + global feature stats
    # ──────────────────────────────────────────────────────────
    print("Pass 1/2: diagnostics + global stats...", flush=True)
    acc          = {"z": {}, "max": {}}
    reports      = []
    target_stats = {}   # asset_id → {"min": ..., "max": ...} after clipping
    excluded     = []   # asset_ids skipped due to degenerate signal or low test coverage
    t0           = time.time()

    for i, row in mapping.iterrows():
        asset_id = row["asset_id"]
        try:
            series = load_measurement(asset_id)

            # ── Exclusion checks ──────────────────────────────
            if is_degenerate(series):
                print(f"  ⚠ EXCLUDED {asset_id}: flat signal (min == max)", flush=True)
                excluded.append((asset_id, "degenerate signal"))
                continue

            if exclude_by_test_nan(series):
                test_pct = round(100 * series[SPLIT_DATE:].isna().mean(), 1)
                print(f"  ⚠ EXCLUDED {asset_id}: test NaN {test_pct}% > {100*(1-MIN_TEST_COVERAGE):.0f}%", flush=True)
                excluded.append((asset_id, f"test NaN {test_pct}%"))
                continue
            # ─────────────────────────────────────────────────

            reports.append(nan_outlier_report(series, asset_id))

            clipped = clip_outliers(series.dropna())
            target_stats[asset_id] = {
                "min": float(clipped.min()),
                "max": float(clipped.max()),
            }

            df_feat = build_nwp_features(
                row["nwp_tile"], row["latitude"], row["longitude"]
            )
            accumulate_stats(df_feat, acc)

        except Exception as e:
            print(f"  ⚠ pass1 skip {asset_id}: {e}", flush=True)

        if (i + 1) % 10 == 0:
            print(f"  stats {i+1}/{len(mapping)} | {time.time()-t0:.0f}s", flush=True)

    # Save NaN/outlier report
    report_df  = pd.DataFrame(reports)
    report_csv = f"{OUT_DIR}/nan_outlier_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"\n--- NaN / Outlier Report (all trafos) ---")
    print(report_df[["asset_id", "n_nan", "pct_nan",
                      "train_nan_pct", "test_nan_pct",
                      "n_outliers_iqr", "pct_outliers",
                      "min_raw", "max_raw"]].to_string(index=False))
    print(f"\nFull report saved → {report_csv}")

    if excluded:
        print(f"\nExcluded {len(excluded)} trafos:")
        for asset_id, reason in excluded:
            print(f"  {asset_id}")

    zstats, maxvals = finalize_stats(acc)
    save_global_stats(zstats, maxvals, target_stats)

    # ──────────────────────────────────────────────────────────
    # PASS 2  Scale features + fill NaN + normalise + save
    # ──────────────────────────────────────────────────────────
    print("\nPass 2/2: scaling + saving...", flush=True)
    scaler          = TrafoFeatureScaler().set_global_stats(zstats, maxvals)
    excluded_ids    = {a for a, _ in excluded}
    t0              = time.time()
    failed          = []

    for i, row in mapping.iterrows():
        asset_id = row["asset_id"]

        if asset_id in excluded_ids:
            continue

        out_path = f"{OUT_DIR}/{asset_id}.parquet"
        if os.path.exists(out_path):
            continue

        try:
            # 1. Target: clip outliers → fill NaN (linear, gap-limited)
            series = load_measurement(asset_id)
            series = clip_outliers(series)
            series = fill_nan_trafo(series, FILL_CONFIG)

            # 2. NWP features: interpolate to 15 min + solar/calendar
            df_feat = build_nwp_features(
                row["nwp_tile"], row["latitude"], row["longitude"]
            )

            # 3. Scale features globally
            df_scaled = scaler.transform(df_feat)

            # 4. Align target to the feature index
            df_out = df_scaled.copy()
            df_out["active_power_MW"] = series.reindex(df_scaled.index)

            # 5. Full-range min-max normalisation → [0, 1]
            #    covers bidirectional signals (negative = net consumption)
            t_min = target_stats[asset_id]["min"]
            t_max = target_stats[asset_id]["max"]
            t_rng = max(t_max - t_min, 1e-8)
            df_out["active_power_MW"] = (
                (df_out["active_power_MW"] - t_min) / t_rng
            ).clip(0.0, 1.0)

            # 6. Save
            df_out.index.name = "t"
            df_out.to_parquet(out_path)

        except Exception as e:
            print(f"  ❌ {asset_id}: {e}", flush=True)
            failed.append(asset_id)
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate    = max((i + 1) / elapsed, 1e-9)
            eta     = (len(mapping) - i - 1) / rate / 60
            print(f"  ✔ {i+1}/{len(mapping)} | {elapsed:.0f}s | ETA {eta:.1f} min",
                  flush=True)

    print(f"\n Done in {(time.time()-t0)/60:.1f} min | failed: {len(failed)}", flush=True)
    if failed:
        print(f"   {failed}")


if __name__ == "__main__":
    run()
