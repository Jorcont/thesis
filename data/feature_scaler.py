import numpy as np
import pandas as pd


class FeatureScaler:
    IRRADIANCE_DIV = 1000.0
    HALF_PI = np.pi / 2

    DIV_CONST = {
        "DIRECT_HORIZONTAL_IRRADIANCE":  IRRADIANCE_DIV,
        "DIFFUSE_HORIZONTAL_IRRADIANCE": IRRADIANCE_DIV,
        "HUMIDITY_RELATIVE":             100.0,
    }

    MINMAX_LB0 = ["WIND_SPEED_40M"]
    ZSCORE_COLS = ["TEMPERATURE_2METER_ABOVE_GROUND", "PRESSURE_SURFACE"]
    AZIMUTH_RAD = ["SOLAR_AZIMUTH"]
    ANGLE_DEG = ["WIND_DEGREES_40M"]
    LEAVE_AS_IS = [
        "TIME_OF_DAY_SIN", "TIME_OF_DAY_COS",
        "DAY_OF_YEAR_SIN", "DAY_OF_YEAR_COS"
    ]
    ELEVATION = "SOLAR_ELEVATION"

    def __init__(self):
        self.zstats = {}
        self.maxvals = {}

    def fit(self, df: pd.DataFrame):
        d = df.dropna()

        for col in self.ZSCORE_COLS:
            if col in d.columns:
                mu = d[col].mean()
                sd = d[col].std()
                self.zstats[col] = (mu, sd if sd > 1e-8 else 1.0)

        for col in self.MINMAX_LB0:
            if col in d.columns:
                mx = d[col].max()
                self.maxvals[col] = mx if mx > 1e-8 else 1.0

        return self

    def set_global_stats(self, zstats: dict, maxvals: dict):
        self.zstats = {k: (float(v[0]), float(v[1])) for k, v in zstats.items()}
        self.maxvals = {k: float(v) for k, v in maxvals.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in df.columns:
            s = df[col]

            if col == self.ELEVATION:
                out[col] = (s / self.HALF_PI).clip(0.0, 1.0)

            elif col in self.AZIMUTH_RAD:
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
                    raise ValueError(f"Missing max value for '{col}'")
                out[col] = (s / self.maxvals[col]).clip(0.0, 1.0)

            elif col in self.ZSCORE_COLS:
                if col not in self.zstats:
                    raise ValueError(f"Missing z-score stats for '{col}'")
                mu, sd = self.zstats[col]
                out[col] = (s - mu) / sd

            elif col in self.LEAVE_AS_IS:
                out[col] = s

            else:
                raise ValueError(f"FeatureScaler: unhandled column '{col}'")

        return out