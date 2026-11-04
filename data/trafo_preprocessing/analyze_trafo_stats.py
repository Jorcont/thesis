import pandas as pd
import numpy as np
import glob
import json

SRC_DIR = "data/trafo_data"
OUT_JSON = "data/trafo_wind_stats.json"

speed_vals = []
sin_vals = []
cos_vals = []

# for rho estimation (lag-1 autocorrelation), collected per file then averaged
speed_rhos = []
angle_rhos = []

files = glob.glob(f"{SRC_DIR}/*.parquet")
print(f"Found {len(files)} trafo files.")

for f in files:
    df = pd.read_parquet(f, columns=["WIND_SPEED_100M", "WIND_DEGREES_100M_SIN", "WIND_DEGREES_100M_COS"])
    df = df.dropna()
    if len(df) < 10:
        continue

    speed = df["WIND_SPEED_100M"].to_numpy()
    sin_c = df["WIND_DEGREES_100M_SIN"].to_numpy()
    cos_c = df["WIND_DEGREES_100M_COS"].to_numpy()

    speed_vals.append(speed)
    sin_vals.append(sin_c)
    cos_vals.append(cos_c)

    # lag-1 autocorrelation for speed (AR(1) persistence estimate)
    if len(speed) > 1:
        s0, s1 = speed[:-1], speed[1:]
        if np.std(s0) > 1e-8 and np.std(s1) > 1e-8:
            speed_rhos.append(np.corrcoef(s0, s1)[0, 1])

    # angle from sin/cos, unwrap-safe autocorrelation via circular difference
    theta = np.arctan2(sin_c, cos_c)
    if len(theta) > 1:
        diff = np.angle(np.exp(1j * (theta[1:] - theta[:-1])))  # wrapped difference
        # concentration of increments ~ persistence: R of increments
        R_inc = np.abs(np.mean(np.exp(1j * diff)))
        angle_rhos.append(R_inc)

speed_all = np.concatenate(speed_vals)
sin_all = np.concatenate(sin_vals)
cos_all = np.concatenate(cos_vals)

speed_mean = float(np.mean(speed_all))
speed_std = float(np.std(speed_all))
speed_min = float(np.min(speed_all))
speed_max = float(np.max(speed_all))

# circular mean + resultant length (concentration proxy) for direction
mean_sin = np.mean(sin_all)
mean_cos = np.mean(cos_all)
R = np.sqrt(mean_sin**2 + mean_cos**2)
mean_angle = float(np.arctan2(mean_sin, mean_cos))

# rough kappa estimate (Fisher approximation), not strictly needed downstream
if R < 0.53:
    kappa = 2 * R + R**3 + 5 * R**5 / 6
elif R < 0.85:
    kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
else:
    kappa = 1 / (R**3 - 4 * R**2 + 3 * R)

rho_speed = float(np.clip(np.mean(speed_rhos), 0.0, 0.999)) if speed_rhos else 0.95
rho_angle = float(np.clip(np.mean(angle_rhos), 0.0, 0.999)) if angle_rhos else 0.95

stats = {
    "speed": {
        "mean": speed_mean,
        "std": speed_std,
        "min": speed_min,
        "max": speed_max,
        "rho": rho_speed,
    },
    "direction": {
        "mean_angle_rad": mean_angle,
        "R": float(R),
        "kappa": float(kappa),
        "rho": rho_angle,
    },
}

with open(OUT_JSON, "w") as fp:
    json.dump(stats, fp, indent=2)

print(json.dumps(stats, indent=2))
print(f"\nSaved stats to {OUT_JSON}")