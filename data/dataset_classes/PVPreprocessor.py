"""
Standalone preprocessing pipeline for PV CSV data.
Extracted from baseline so it can be reused here.

Potentially, this should change when importing
the module given in the PylOrin repository. --> task for later
"""

import os
import pandas as pd
import numpy as np


class PVPreprocessor:
    """
    Loads, interpolates, aligns and merges features + target CSVs.

    Additionally adds:
        - time_idx     : integer step counter (required by TFT / TimeSeriesDataSet)
        - plant_id     : plant identifier extracted from filename (group_id for TFT)
        - Cyclic calendar features derived from start_time_utc:
              hour_sin, hour_cos  → time-of-day
              day_sin,  day_cos   → day-of-year
              month               → calendar month
          These are time_varying_known_reals. Calendar dates are always known in advance.
    """

    TIME_COLUMN      = "start_time_utc"
    TARGET_FREQUENCY = "15min"

    FEATURE_COLUMNS = [
        "GLOBAL_HORIZONTAL_IRRADIANCE",
        "TEMPERATURE_2METER_ABOVE_GROUND",
        "WIND_SPEED_40M",
    ]
    TARGET_COLUMNS = ["target"]

    # Calendar columns added by _add_time_features() --> See below
    CALENDAR_COLUMNS = [
        "time_idx",
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month",
    ]

    def __init__(
        self,
        features_csv_path: str, # path to be provided
        target_csv_path:   str, # path to be provided
        interpolate: bool = True,
        align_time:  bool = True,
    ):
        self.features_csv_path = features_csv_path
        self.target_csv_path   = target_csv_path
        self.interpolate       = interpolate
        self.align_time        = align_time
        self._df               = None

        # Extract plant name from filename → used as group_id for TFT
        self._plant_name = (
            os.path.basename(features_csv_path)
            .replace("pattern_", "")
            .replace(".csv", "")
        )

    # Get the DataFrame
    def get_dataframe(self) -> pd.DataFrame:
        """Run pipeline and return merged DataFrame (feat + target)."""
        if self._df is None:
            self._df = self._build()
        return self._df

    # ──────────────────────────────────────────────────────────────
    #  Pipeline steps
    # ──────────────────────────────────────────────────────────────

    def _build(self) -> pd.DataFrame:
        features_df = self._load(self.features_csv_path)
        target_df   = self._load(self.target_csv_path)

        if self.interpolate:
            # Only interpolate the features
            features_df = self._interpolate(features_df)
        if self.align_time:
            features_df, target_df = self._align(features_df, target_df)

        merged = pd.merge(
            features_df[[self.TIME_COLUMN] + self.FEATURE_COLUMNS],
            target_df[[self.TIME_COLUMN]   + self.TARGET_COLUMNS],
            on=self.TIME_COLUMN, how="inner",
        ).sort_values(self.TIME_COLUMN).reset_index(drop=True)

        merged = merged.ffill().bfill()   # Remove and fill the NaN
        #print(merged.head())

        # New: add TFT-required cyclic columns
        merged = self._add_time_features(merged)
        #print(merged.head())
        return merged

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)    # We need the path of the dataset
        df[self.TIME_COLUMN] = pd.to_datetime(df[self.TIME_COLUMN], utc=True)
        return df.sort_values(self.TIME_COLUMN).reset_index(drop=True)

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the interpolated (linearly) dataframe w.r.t time.
        """
        return (
            df.set_index(self.TIME_COLUMN)
              .resample(self.TARGET_FREQUENCY)
              .interpolate(method="time")   # linear interpolation, w.r.t time
              .reset_index()
        )

    def _align(self, features_df, target_df):
        """Returns a DataFrame where both all the information is included."""
        t_f   = features_df[self.TIME_COLUMN]
        t_t   = target_df[self.TIME_COLUMN]
        start = max(t_f.min(), t_t.min())   # look for the maximum
        end   = min(t_f.max(), t_t.max())   # look for the minimum
        return (
            features_df[(t_f >= start) & (t_f <= end)].reset_index(drop=True),
            target_df  [(t_t >= start) & (t_t <= end)].reset_index(drop=True),
        )

    # ──────────────────────────────────────────────────────────────
    # TFT-specific feature engineering
    # ──────────────────────────────────────────────────────────────

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns required by TimeSeriesDataConfig / TFT:

        time_idx  — integer step counter (mandatory for TimeSeriesDataSet)
        plant_id  — plant identifier extracted from filename (group_id)

        Cyclic calendar features (time_varying_known_reals):
            hour_sin / hour_cos  → smooth encoding of time-of-day (15-min res)
            day_sin  / day_cos   → smooth encoding of day-of-year
            month                → calendar month (1–12)

        Why cyclic?  A linear hour column jumps from 23 → 0 at midnight,
        creating a false discontinuity.  sin/cos encoding wraps smoothly.
        """
        # df = df.copy()

        # ── Mandatory TFT index ───────────────────────────────────
        df["time_idx"] = df.index.astype(int)

        # ── Plant identifier (static, used as group_id) ───────────
        df["plant_id"] = self._plant_name

        # ── Parse timestamps ──────────────────────────────────────
        ts = pd.to_datetime(df[self.TIME_COLUMN]) # Returns DataSeries
        #print(ts.head())
        #print(ts.shape)

        # Fractional hour  →  0.0 ... 23.75  for 15-min data
        # Convert into decimal hours --> for enconding
        hour_frac      = ts.dt.hour + ts.dt.minute / 60.0
        # print(hour_frac)
        day_of_year    = ts.dt.dayofyear.astype(float)
        # print(day_of_year)

        # ── Cyclic encodings ──────────────────────────────────────
        df["hour_sin"] = np.sin(2 * np.pi * hour_frac   / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac   / 24.0)
        df["day_sin"]  = np.sin(2 * np.pi * day_of_year / 365.0)
        df["day_cos"]  = np.cos(2 * np.pi * day_of_year / 365.0)
        df["month"]    = ts.dt.month.astype(float)

        return df