import pandas as pd
import numpy as np
import glob
import os
import json
import time
import hashlib

SRC_DIR = "data/PV_data"
DST_DIR = "data/PV_data_trafo_random"
STATS_JSON = "data/trafo_wind_stats.json"
os.makedirs(DST_DIR, exist_ok=True)

with open(STATS_JSON) as fp:
    STATS = json.load(fp)

SPEED_MEAN = STATS["speed"]["mean"]
SPEED_STD = STATS["speed"]["std"]
SPEED_RHO = STATS["speed"]["rho"]

ANGLE_MEAN = STATS["direction"]["mean_angle_rad"]
ANGLE_RHO = STATS["direction"]["rho"]
# angular noise scale derived from concentration R -> circular std approx
R = STATS["direction"]["R"]
ANGLE_STD = np.sqrt(-2 * np.log(max(R, 1e-6)))  # circular std (radians), capped

FINAL_COLS = [
    'WIND_DEGREES_40M_SIN', 'WIND_DEGREES_40M_COS', 'PRESSURE_SURFACE',
    'DIRECT_HORIZONTAL_IRRADIANCE', 'WIND_SPEED_40M',
    'TEMPERATURE_2METER_ABOVE_GROUND', 'HUMIDITY_RELATIVE',
    'DIFFUSE_HORIZONTAL_IRRADIANCE', 'SOLAR_ELEVATION',
    'SOLAR_AZIMUTH_SIN', 'SOLAR_AZIMUTH_COS',
    'TIME_OF_DAY_SIN', 'TIME_OF_DAY_COS',
    'DAY_OF_YEAR_SIN', 'DAY_OF_YEAR_COS',
    'WIND_DEGREES_100M_SIN', 'WIND_DEGREES_100M_COS', 'WIND_SPEED_100M',
    'active_power_kW'
]


def seeded_rng(fname: str) -> np.random.Generator:
    """Deterministic per-file seed for reproducibility."""
    h = int(hashlib.md5(fname.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(h)


def generate_speed_series(n: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1)/OU process for wind speed, clipped to [0, 1]."""
    innov_std = SPEED_STD * np.sqrt(max(1 - SPEED_RHO**2, 1e-8))
    x = np.empty(n)
    x[0] = rng.normal(SPEED_MEAN, SPEED_STD)
    for t in range(1, n):
        x[t] = SPEED_MEAN + SPEED_RHO * (x[t - 1] - SPEED_MEAN) + rng.normal(0, innov_std)
    return np.clip(x, 0.0, 1.0)


def generate_direction_series(n: int, rng: np.random.Generator):
    """AR(1) process on the angle (radians), wrapped, then converted to sin/cos.
    Guarantees sin^2 + cos^2 = 1 exactly."""
    innov_std = ANGLE_STD * np.sqrt(max(1 - ANGLE_RHO**2, 1e-8))
    theta = np.empty(n)
    theta[0] = ANGLE_MEAN + rng.normal(0, ANGLE_STD)
    for t in range(1, n):
        delta = ANGLE_RHO * np.angle(np.exp(1j * (theta[t - 1] - ANGLE_MEAN)))
        theta[t] = ANGLE_MEAN + delta + rng.normal(0, innov_std)
    theta = np.angle(np.exp(1j * theta))  # wrap to [-pi, pi]
    return np.sin(theta), np.cos(theta)


files = glob.glob(f"{SRC_DIR}/*.parquet")
n_files = len(files)
failed = []
t0 = time.time()

print(f"Processing {n_files} files...")

for i, f in enumerate(files):
    fname = os.path.basename(f)
    out = f"{DST_DIR}/{fname}"

    if os.path.exists(out):
        continue

    try:
        df = pd.read_parquet(f)
        n = len(df)
        rng = seeded_rng(fname)

        df["WIND_SPEED_100M"] = generate_speed_series(n, rng)
        sin_series, cos_series = generate_direction_series(n, rng)
        df["WIND_DEGREES_100M_SIN"] = sin_series
        df["WIND_DEGREES_100M_COS"] = cos_series

        df[FINAL_COLS].to_parquet(out)
    except Exception as e:
        failed.append((fname, str(e)))

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        eta = (n_files - i - 1) / ((i + 1) / elapsed) / 60
        print(f"  {i+1}/{n_files} | {elapsed:.0f}s | ETA {eta:.1f} min")

print(f"\nDone in {(time.time()-t0)/60:.1f} min | failed: {len(failed)}")
if failed:
    for fname, err in failed:
        print(f"  {fname}: {err}")