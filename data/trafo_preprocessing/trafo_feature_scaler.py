"""
Feature scaler for trafo preprocessing.
Extends the original FeatureScaler with WIND_SPEED_100M / WIND_DEGREES_100M.
Logic is identical to feature_scaler.py – only the column lists are extended.
"""

import numpy as np
import pandas as pd


class TrafoFeatureScaler:
    """
    Deterministic, stateless transformations + fitted z-score / min-max params.

    Column routing
    --------------
    SOLAR_ELEVATION              → divide by π/2, clip [0, 1]
    SOLAR_AZIMUTH                → sin/cos (radians, from BuildFeatures)
    WIND_DEGREES_{40,100}M       → deg2rad → sin/cos
    DIRECT/DIFFUSE_*_IRRADIANCE  → divide by 1 000, clip ≥ 0
    HUMIDITY_RELATIVE            → divide by 100, clip ≥ 0
    WIND_SPEED_{40,100}M         → min-max via global max, clip [0, 1]
    TEMPERATURE_2METER / PRESSURE_SURFACE → z-score (global mean/std)
    TIME_OF_DAY_{SIN,COS}        → as-is  (already in [-1, 1])
    DAY_OF_YEAR_{SIN,COS}        → as-is
    """

    HALF_PI = np.pi / 2

    DIV_CONST = {
        "DIRECT_HORIZONTAL_IRRADIANCE":  1_000.0,
        "DIFFUSE_HORIZONTAL_IRRADIANCE": 1_000.0,
        "HUMIDITY_RELATIVE":               100.0,
    }
    MINMAX_LB0  = ["WIND_SPEED_40M",     "WIND_SPEED_100M"]
    ZSCORE_COLS = ["TEMPERATURE_2METER_ABOVE_GROUND", "PRESSURE_SURFACE"]
    AZIMUTH_RAD = ["SOLAR_AZIMUTH"]
    ANGLE_DEG   = ["WIND_DEGREES_40M",   "WIND_DEGREES_100M"]
    LEAVE_AS_IS = [
        "TIME_OF_DAY_SIN", "TIME_OF_DAY_COS",
        "DAY_OF_YEAR_SIN", "DAY_OF_YEAR_COS",
    ]
    ELEVATION = "SOLAR_ELEVATION"

    def __init__(self):
        self.zstats:  dict = {}   # col -> (mean, std)
        self.maxvals: dict = {}   # col -> global_max

    # ── Fitting (local, single-trafo) ─────────────────────────
    def fit(self, df: pd.DataFrame) -> "TrafoFeatureScaler":
        d = df.dropna()
        for col in self.ZSCORE_COLS:
            if col in d.columns:
                mu, sd = d[col].mean(), d[col].std()
                self.zstats[col] = (float(mu), float(sd if sd > 1e-8 else 1.0))
        for col in self.MINMAX_LB0:
            if col in d.columns:
                mx = d[col].max()
                self.maxvals[col] = float(mx if mx > 1e-8 else 1.0)
        return self

    # ── Load pre-computed global stats ────────────────────────
    def set_global_stats(self, zstats: dict,
                         maxvals: dict) -> "TrafoFeatureScaler":
        self.zstats  = {k: (float(v[0]), float(v[1]))
                        for k, v in zstats.items()}
        self.maxvals = {k: float(v) for k, v in maxvals.items()}
        return self

    # ── Transform ─────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in df.columns:
            s = df[col]

            if col == self.ELEVATION:
                out[col] = (s / self.HALF_PI).clip(0.0, 1.0)

            elif col in self.AZIMUTH_RAD:
                # BuildFeatures outputs azimuth in radians
                out[f"{col}_SIN"] = np.sin(s)
                out[f"{col}_COS"] = np.cos(s)

            elif col in self.ANGLE_DEG:
                rad = np.deg2rad(s)
                out[f"{col}_SIN"] = np.sin(rad)
                out[f"{col}_COS"] = np.cos(rad)

            elif col in self.DIV_CONST:
                out[col] = (s / self.DIV_CONST[col]).clip(lower=0.0)

            elif col in self.MINMAX_LB0:
                if col not in self.maxvals:
                    raise ValueError(f"Missing maxval for '{col}'")
                out[col] = (s / self.maxvals[col]).clip(0.0, 1.0)

            elif col in self.ZSCORE_COLS:
                if col not in self.zstats:
                    raise ValueError(f"Missing z-score stats for '{col}'")
                mu, sd = self.zstats[col]
                out[col] = (s - mu) / sd

            elif col in self.LEAVE_AS_IS:
                out[col] = s

            else:
                raise ValueError(
                    f"TrafoFeatureScaler: unhandled column '{col}'. "
                    f"Add it to the appropriate list.")

        return out
