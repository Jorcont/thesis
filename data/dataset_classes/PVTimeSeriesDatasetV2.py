"""
PVTimeSeriesDatasetV2 — Arrow pre-load strategy (Strategy C benchmark winner)

Architecture:
    Job start → pre-load all plants as Arrow tables → RAM (very big on CLUSTER)
    Training  → O(1) zero-copy Arrow slice per window
    
Interface (unchanged for MAML pipeline):
    LSTM: (X_support, y_support, X_query, y_query)
    TFT : (X_enc_s, X_dec_s), y_s, (X_enc_q, X_dec_q), y_q
    TFT (pytorch_forecasting): dict with each part of the network
"""
import random
import time
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any, Optional
from typing_extensions import override
from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from algorithms.PipelineState import PipelineState as state

class PVTimeSeriesDatasetV2(SupervisedLearningDataset):
    """
    PV time series dataset backed by Arrow in-memory tables.
    All plants are pre-loaded once at job start via PVTaskSamplerV2 into a class-level (i.e. shared across all Dataset instances/splits) Arrow cache held in RAM. 
    Window sampling support/query is O(1) zero-copy Arrow slice → numpy conversion → torch.
    Supports three downstream model interfaces (selected via config flags
    set in ``init_data``):
        - LSTM  : plain (X, y) tensors
        - TFT   : custom encoder/decoder split (this project's own TFT)
        - PF-TFT: dict format expected by the `pytorch_forecasting` library

    Pool format:
        List of plant_ids (strings)  ["51444188374", "50176122064", ...]
    This is the set of plants assigned to this particular dataset instance (e.g. training split, validation split, or test split)
    """

    # ── Class-level Arrow cache (shared across train/val/test) ─
    # Class attributes. This means the cache is populated ONCE per process (via 'preload') and then shared by every PVTimeSeriesDatasetV2 instance (train/val/test splits alike), avoiding redundant loading of the same plant data into memory multiple times.
    _arrow_cache: dict = {}   # { plant_id: pa.Table }
    _rated_power: dict = {}   # { plant_id: float (kW) } - cap_sum, used for de-normalization
    _static_features: dict = {}  # {plant_id: np.ndarray([lat_norm, lon_norm])}
    _timestamps: dict = {} # { plant_id: np.ndarray[datetime64] } - row aligned index
    # ── Column definitions ────────────────────────────────────
    # Input features availalbe to the encoder (past window). 
    # These are the normalized/engineerd NWP + solar position + calendar features.
    FEATURE_COLS: List[str] = [
        "WIND_DEGREES_40M_SIN",
        "WIND_DEGREES_40M_COS",
        "PRESSURE_SURFACE",
        "DIRECT_HORIZONTAL_IRRADIANCE",
        "WIND_SPEED_40M",
        "TEMPERATURE_2METER_ABOVE_GROUND",
        "HUMIDITY_RELATIVE",
        "DIFFUSE_HORIZONTAL_IRRADIANCE",
        "SOLAR_ELEVATION",
        "SOLAR_AZIMUTH_SIN",
        "SOLAR_AZIMUTH_COS",
        "TIME_OF_DAY_SIN",
        "TIME_OF_DAY_COS",
        "DAY_OF_YEAR_SIN",
        "DAY_OF_YEAR_COS",
    ]
    TARGET_COL: str = "active_power_kW"

    # ── CHANGE 1: add trafo extended feature cols ─────────────────
    TRAFO_FEATURE_COLS: List[str] = FEATURE_COLS + [
        "WIND_DEGREES_100M_SIN",
        "WIND_DEGREES_100M_COS",
        "WIND_SPEED_100M",
    ]

    # TFT decoder — only future-known features (solar + time)
    DEC_FEATURE_COLS: List[str] = [
        "SOLAR_ELEVATION",
        "SOLAR_AZIMUTH_SIN",
        "SOLAR_AZIMUTH_COS",
        "TIME_OF_DAY_SIN",
        "TIME_OF_DAY_COS",
        "DAY_OF_YEAR_SIN",
        "DAY_OF_YEAR_COS",
    ]
    ALL_COLS: List[str] = FEATURE_COLS + [TARGET_COL]

    # ──────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────
    @override
    def init_data(self):
        """Initialise dataset from DataConfig (YAML).
        Reads window geometry (look_back/look_ahead/window_stride/gap), the support/query split ratio, per-split sample caps, and the day-validity filtering thresholds used later in 'sample_window'. 
        Also pre-computes integer column indeces for feature/target/decoder"""
        from conf.data_configs import DataConfig
        data_cfg = DataConfig.get_config()

        cfg = data_cfg.config if data_cfg is not None else {}
        # ── Window params ─────────────────────────────────────
        self.look_back_steps  = cfg.get("look_back_steps", 96) # encoder/history length
        self.look_ahead_steps = cfg.get("look_ahead_steps", 96) # forecast horizon
        self.window_steps     = self.look_back_steps + self.look_ahead_steps 
        self.window_stride    = cfg.get("window_stride", 1) # step size between consecutive windows
        self.train_ratio      = cfg.get("train_ratio", 0.7) # support vs. query split ratio
        self.max_windows      = cfg.get("max_windows", None)
        self.gap              = cfg.get("gap", 96) # temporal buffer between support and query (avoid evaluating in seen data, also known as overlapping)
        self.random_sampling  = cfg.get("random_sampling", False) # antoher way of sampling
        if self.gap < self.look_ahead_steps:
            raise ValueError(f"The gap: {self.gap} between the support and query sets\n" 
                             f"must be >= {self.look_ahead_steps}: look ahead window\n"
                             f"This avoids DATA LEAKAGE and overlapping of sets")

        # ── Separate caps for train vs test ───────────────────
        self.max_windows_train = cfg.get("max_windows_train", 500)
        self.max_windows_val   = cfg.get("max_windows_val", self.max_windows_train)
        self.max_windows_test  = cfg.get("max_windows_test", 500)

        # ── TFT mode detection ────────────────────────────────
        # The presence of "time_varying_known_reals" in the config is used as an implicit flag to switch the whole packaging pipeline into TFT mode.
        self.use_tft    = "time_varying_known_reals" in cfg
        self.use_pf_tft = cfg.get("use_pf_tft", False)

        # ── CHANGE 2: in init_data — detect flag, set instance-level active cols ──
        use_trafo = cfg.get("use_trafo_features", False)      
        self._active_feat_cols = (
            self.TRAFO_FEATURE_COLS if use_trafo else self.FEATURE_COLS
        )
        self._active_all_cols = self._active_feat_cols + [self.TARGET_COL]

        # Fixed seed base for validation/test sampling, so that repeated
        # evaluation runs (e.g. across epochs) draw the SAME windows,
        # making metrics comparable over time.
        self._is_val  = False
        self._is_test = False
        self._eval_seed = random.randint(0, 99_999)

        # ── Pre-compute column indices (avoid per-call string lookups) ─
        self._feat_idx   = [self._active_all_cols.index(c) for c in self._active_feat_cols]
        self._target_idx = self._active_all_cols.index(self.TARGET_COL)
        self._dec_idx    = [self._active_all_cols.index(c) for c in self.DEC_FEATURE_COLS]

        # ── Day-validity filter (TRAIN only) ──────────────────
        # These parameters control the rejection-sampling scheme in 'sample_window', which discards windows/palnts whose target series contains too "easy" days (e.g. long streches of zero power)
        self.DAY_STEPS       = 96
        self.MIN_NNZ_DAY     = cfg.get("min_nnz_day", 18)      # 15–20, winter-safe
        self.THETA_SUPPORT   = cfg.get("theta_support", 0.75)   # strict
        self.THETA_QUERY     = cfg.get("theta_query", 0.75) # strict
        self.MAX_RESAMPLE    = cfg.get("max_resample", 5)
        self.MAX_PLANT_SWITCH= cfg.get("max_plant_switch", 20)
        self.n_rejected_windows = 0    # diagnostics only
        self.n_rejected_plants  = 0

        print(
            f"[PVTimeSeriesDatasetV2] "
            f"mode={'PF-TFT' if self.use_pf_tft else 'TFT' if self.use_tft else 'LSTM'} | "
            f"look_back={self.look_back_steps} | "
            f"look_ahead={self.look_ahead_steps} | "
            f"stride={self.window_stride} | "
            f"max_windows: train={self.max_windows_train}, test={self.max_windows_test} | "
            f"eval_seed={self._eval_seed}"
        )

    # ──────────────────────────────────────────────────────────
    #  Pre-load (called once by PVTaskSamplerV2)
    # ──────────────────────────────────────────────────────────
    @classmethod
    def preload(cls, plant_ids: List[str], plants_dir: str,
                n_workers: int = 32,
                columns: Optional[List[str]] = None) -> None:
        """Load all plants as Arrow tables into class-level cache."""
        to_load = [p for p in plant_ids if p not in cls._arrow_cache]
        if not to_load:
            print(f"[PVTimeSeriesDatasetV2] Cache already warm "
                  f"({len(cls._arrow_cache)} plants)")
            return

        read_cols = columns if columns is not None else cls.ALL_COLS 


        print(f"[PVTimeSeriesDatasetV2] Pre-loading {len(to_load)} plants ...")
        t0 = time.perf_counter()

        def _load(plant_id):
            """Reads one plant's parquet file into an Arrow table, forward/backward-fills
            NaNs in the target column, and captures the plant's row-aligned datetime index."""
            path  = f"{plants_dir}/{plant_id}.parquet"
            table = pq.read_table(path, columns=read_cols)
            col_idx = table.schema.get_field_index(cls.TARGET_COL)
            arr = table.column(col_idx).to_pandas()
            arr = arr.ffill().bfill().values.astype(np.float32)
            table = table.set_column(
                col_idx, cls.TARGET_COL,
                pa.array(arr, type=pa.float32())
            )
            # capture datetime index (index-only read is fast) ──
            ts_df = pd.read_parquet(path, columns=[])
            cls._timestamps[plant_id] = (
                ts_df.index.tz_localize(None).values   
                if ts_df.index.tz else ts_df.index.values
            )
            # ──────────────────────────────────────────────────────────
            return plant_id, table

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for plant_id, table in pool.map(_load, to_load):
                cls._arrow_cache[plant_id] = table

        elapsed = time.perf_counter() - t0
        print(f"[PVTimeSeriesDatasetV2] ✅ Loaded {len(to_load)} plants "
              f"in {elapsed:.2f}s | Cache total: {len(cls._arrow_cache)}")

    @classmethod
    def load_rated_powers(cls, plants_dir: str) -> None:
        """Load cap_sum per plant from _renorm_report.csv."""
        if cls._rated_power:
            return

        report_path = os.path.join(plants_dir, "_renorm_report.csv")
        if not os.path.exists(report_path):
            print(f"[PVTimeSeriesDatasetV2] ⚠️  _renorm_report.csv not found")
            return

        t0 = time.perf_counter()
        df = pd.read_csv(report_path, dtype={"plant_id": str})
        for _, row in df.iterrows():
            plant_id = str(row["plant_id"]).strip()
            cap_sum = row["cap_sum"]
            if plant_id and not pd.isna(cap_sum) and cap_sum > 0:
                cls._rated_power[plant_id] = float(cap_sum)

        elapsed = time.perf_counter() - t0
        print(f"[PVTimeSeriesDatasetV2] ✅ Loaded rated powers for "
              f"{len(cls._rated_power)} plants ({elapsed:.2f}s)")

        # Consider Removing this. Everything is working well. DEBUG
        missing = set(cls._arrow_cache.keys()) - set(cls._rated_power.keys())
        if missing:
            print(f"  ⚠️  {len(missing)} cached plants have NO cap_sum")

    @classmethod
    def load_static_features(cls, mapping_path: str) -> None:
        """Load normalised lat/lon per plant."""
        if cls._static_features:
            return

        df = pd.read_parquet(mapping_path)
        df["asset_id"] = df["asset_id"].astype(str).str.strip()

        lat_mean, lat_std = df["LATITUDE"].mean(), df["LATITUDE"].std()
        lon_mean, lon_std = df["LONGITUDE"].mean(), df["LONGITUDE"].std()

        for _, row in df.iterrows():
            pid = str(row["asset_id"]).strip()
            cls._static_features[pid] = np.array([
                (row["LATITUDE"]  - lat_mean) / (lat_std  + 1e-8),
                (row["LONGITUDE"] - lon_mean) / (lon_std  + 1e-8),
            ], dtype=np.float32)

        print(f"[PVTimeSeriesDatasetV2] ✅ Static features loaded for "
              f"{len(cls._static_features)} plants")

    # ──────────────────────────────────────────────────────────
    #  Entry point
    # ──────────────────────────────────────────────────────────
    @override
    def __getitem__(self, idx):
        return self.sample_window(idx)

    @override
    def __len__(self):
        """Number of plants/tasks available in the split's pool"""
        return len(self.pool) 

    # ──────────────────────────────────────────────────────────
    #  Core sampling
    # ──────────────────────────────────────────────────────────
    def _sample_once(self, plant_id, idx) -> Tuple:
        """Samples one support/query split for a given plant (no day-validity
        rejection). Used for deterministic val/test sampling and for the
        random-sampling training path. Computes the split sizes/gap via
        `_compute_split`, picks a (seeded, for val/test) start offset, and
        dispatches to `_build_task`.
        """
        table  = self._arrow_cache[plant_id]
        n_rows = table.num_rows

        self._last_plant_id       = plant_id
        self._last_rated_power_kW = self._rated_power.get(plant_id, None)

        n_windows = (n_rows - self.window_steps) // self.window_stride + 1
        min_needed = self._compute_min_windows()
        if n_windows < min_needed:
            raise ValueError(
                f"Plant {plant_id}: n_windows={n_windows} < min_needed={min_needed}")

        n_support, n_query, gap, total_needed = self._compute_split(n_windows)
        max_start = n_windows - total_needed

        if self._is_val or self._is_test:
            rng = random.Random(self._eval_seed + idx)
            start_idx = rng.randint(0, max(0, max_start))
        else:
            start_idx = random.randint(0, max(0, max_start))

        start_row       = start_idx * self.window_stride
        query_start_row = start_row + (n_support * self.window_stride) + gap

        return self._build_task(table, plant_id, start_row, n_support, n_query, query_start_row)
    


    # ──────────────────────────────────────────────────────────
    #  Unified task builder (single dispatch point)
    # ──────────────────────────────────────────────────────────
    def _build_task(
        self,
        table: pa.Table,
        plant_id: str,
        start_row: int,
        n_support: int,
        n_query: int,
        query_start_row: int,
    ) -> Tuple:
        """
        Build support/query task in the appropriate format.
        Dispatches to LSTM, custom TFT, or pytorch_forecasting TFT
        based on config flags.
        """        

        # Always use TRAFO_FEATURE_COLS — all data (PV + trafo) has 18 features
        feat_idx   = [self._active_all_cols.index(c) for c in self._active_feat_cols]
        target_idx = self._active_all_cols.index(self.TARGET_COL)
        dec_idx    = [self._active_all_cols.index(c) for c in self.DEC_FEATURE_COLS]

        # ── Extract raw windows for support & query ───────────
        windows_s = self._extract_windows(table, start_row, n_support)
        windows_q = self._extract_windows(table, query_start_row, n_query)

        # ── NEW ───────────────────────────────────────────────────
        self._set_timestamps(plant_id, start_row, n_support, n_query, query_start_row)
        # ─────────────────────────────────────────────────────────
        # ── Split & package per mode ──────────────────────────
        if self.use_pf_tft:
            return self._package_pf_tft(
                windows_s, windows_q, feat_idx, dec_idx, target_idx, plant_id)
        elif self.use_tft:
            return self._package_tft(
                windows_s, windows_q, feat_idx, dec_idx, target_idx)
        else:
            return self._package_lstm(
                windows_s, windows_q, feat_idx, target_idx)

    # ──────────────────────────────────────────────────────────
    #  Core windowing (SINGLE implementation — no more copies)
    # ──────────────────────────────────────────────────────────
    def _extract_windows(
        self,
        table: pa.Table,
        start_row: int,
        n_windows: int,
    ) -> np.ndarray:
        """
        Vectorized strided sliding-window extraction.

        Converts Arrow → numpy ONCE, then uses broadcast indexing
        to gather all windows without a Python loop.

        Args:
            table     : Arrow table for the plant
            start_row : first row offset (already in row-space, not window-index)
            n_windows : number of windows to extract

        Returns:
            windows : [N, window_steps, n_cols] float32
                      Raw windowed data — column splitting is caller's job.
                      Returns shape [0, window_steps, n_cols] if n_windows == 0.
        """
        win    = self.window_steps
        stride = self.window_stride
        # n_cols = len(self._active_all_cols)
        n_cols = table.num_columns

        if n_windows == 0:
            return np.zeros((0, win, n_cols), dtype=np.float32)

        # Total rows spanned by n_windows with given stride
        total_rows = (n_windows - 1) * stride + win
        chunk = table.slice(start_row, total_rows)
        arr   = chunk.to_pandas().values.astype(np.float32)
        # arr shape: [total_rows, n_cols]

        # Broadcast index: window i starts at row i*stride
        row_idx = (np.arange(n_windows)[:, None] * stride   # [N, 1]
                   + np.arange(win)[None, :])                # [1, W]
        # row_idx shape: [N, W]

        return arr[row_idx]  # [N, W, n_cols]

    def _filter_nan(self, *arrays) -> Tuple[np.ndarray, ...]:
        """
        Remove windows that contain ANY NaN across all provided arrays.

        Args:
            *arrays: variable number of arrays, all with same first dim N.

        Returns:
            Filtered arrays (same order), keeping only rows valid across ALL.
        """
        valid = np.ones(arrays[0].shape[0], dtype=bool)
        for arr in arrays:
            valid &= ~np.isnan(arr).reshape(arr.shape[0], -1).any(axis=1)

        return tuple(arr[valid] for arr in arrays)

    # ──────────────────────────────────────────────────────────
    #  Mode-specific packaging (thin — only column logic)
    # ──────────────────────────────────────────────────────────
    def _package_lstm(
        self,
        windows_s: np.ndarray,
        windows_q: np.ndarray,
        feat_idx: List[int],
        target_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LSTM format:
            X [N, look_back,  n_features]
            y [N, look_ahead, 1]
        """
        lb = self.look_back_steps

        def _split(windows):
            X = windows[:, :lb, :][:, :, feat_idx]        # [N, lb, F]
            y = windows[:, lb:, :][:, :, [target_idx]]    # [N, la, 1]
            X, y = self._filter_nan(X, y)
            return X, y

        X_s, y_s = _split(windows_s)
        X_q, y_q = _split(windows_q)

        return (
            torch.from_numpy(X_s),
            torch.from_numpy(y_s),
            torch.from_numpy(X_q),
            torch.from_numpy(y_q),
        )

    def _package_tft(
        self,
        windows_s: np.ndarray,
        windows_q: np.ndarray,
        feat_idx: List[int],
        dec_idx: List[int],
        target_idx: int,
    ) -> Tuple:
        """
        Custom TFT format:
            (X_enc [N, lb, F_enc], X_dec [N, la, F_dec]), y [N, la, 1]
        """
        lb = self.look_back_steps

        def _split(windows):
            X_enc = windows[:, :lb, :][:, :, feat_idx]     # [N, lb, F_enc]
            X_dec = windows[:, lb:, :][:, :, dec_idx]      # [N, la, F_dec]
            y     = windows[:, lb:, :][:, :, [target_idx]] # [N, la, 1]
            X_enc, X_dec, y = self._filter_nan(X_enc, X_dec, y)
            return X_enc, X_dec, y

        Xe_s, Xd_s, y_s = _split(windows_s)
        Xe_q, Xd_q, y_q = _split(windows_q)

        return (
            (torch.from_numpy(Xe_s), torch.from_numpy(Xd_s)),
            torch.from_numpy(y_s),
            (torch.from_numpy(Xe_q), torch.from_numpy(Xd_q)),
            torch.from_numpy(y_q),
        )

    def _package_pf_tft(
        self,
        windows_s: np.ndarray,
        windows_q: np.ndarray,
        feat_idx: List[int],
        dec_idx: List[int],
        target_idx: int,
        plant_id: str,
    ) -> Tuple:
        """
        pytorch_forecasting TFT dict format.
        Encoder cont includes target as past-observable feature.
        Decoder cont uses ALL known reals.
        """
        lb = self.look_back_steps

        def _split(windows):
            enc_features = windows[:, :lb, :][:, :, feat_idx]    # [N, lb, 13]
            enc_target   = windows[:, :lb, :][:, :, [target_idx]]  # [N, lb, 1]
            X_enc = np.concatenate([enc_features, enc_target], axis=-1)  # [N, lb, 14]
            X_dec = windows[:, lb:, :][:, :, feat_idx]           # [N, la, 13]
            y     = windows[:, lb:, :][:, :, [target_idx]]       # [N, la, 1]
            X_enc, X_dec, enc_target, y = self._filter_nan(X_enc, X_dec, enc_target, y)
            return X_enc, X_dec, enc_target, y

        Xe_s, Xd_s, enc_tgt_s, y_s = _split(windows_s)
        Xe_q, Xd_q, enc_tgt_q, y_q = _split(windows_q)

        x_dict_s = self._to_pf_dict(Xe_s, Xd_s, enc_tgt_s, plant_id=plant_id)
        x_dict_q = self._to_pf_dict(Xe_q, Xd_q, enc_tgt_q, plant_id=plant_id)

        return (
            x_dict_s,
            torch.from_numpy(y_s),
            x_dict_q,
            torch.from_numpy(y_q),
        )

    # ──────────────────────────────────────────────────────────
    #  pytorch_forecasting dict builder
    # ──────────────────────────────────────────────────────────
    def _to_pf_dict(
        self,
        X_enc: np.ndarray,
        X_dec: np.ndarray,
        enc_target: np.ndarray,
        plant_id: Optional[str] = None,
    ) -> dict:
        """
        Convert numpy arrays → pytorch_forecasting TFT input dict.

        Column layout (16 total):
            [0-1]  static: LATITUDE, LONGITUDE
            [2-14] time_varying_known_reals (13 features)
            [15]   time_varying_unknown_reals (active_power_kW)
        """
        B  = X_enc.shape[0]
        lb = self.look_back_steps
        la = self.look_ahead_steps

        # ── Static reals → [B, T, 2] ─────────────────────────
        if plant_id is not None and plant_id in self._static_features:
            static_vec = torch.tensor(self._static_features[plant_id])
        else:
            static_vec = torch.zeros(2)

        static_enc = static_vec[None, None, :].expand(B, lb, 2).contiguous()
        static_dec = static_vec[None, None, :].expand(B, la, 2).contiguous()

        # ── Continuous tensors (16 columns) ───────────────────
        encoder_cont = torch.cat([
            static_enc,                       # [B, lb, 2]
            torch.from_numpy(X_enc),          # [B, lb, 14]
        ], dim=-1)                            # [B, lb, 16]

        decoder_cont = torch.cat([
            static_dec,                       # [B, la, 2]
            torch.from_numpy(X_dec),          # [B, la, 13]
            torch.zeros(B, la, 1),            # [B, la, 1]  target placeholder
        ], dim=-1)                            # [B, la, 16]

        return {
            "encoder_cont":     encoder_cont,
            "decoder_cont":     decoder_cont,
            "encoder_target":   torch.from_numpy(enc_target).squeeze(-1),
            "decoder_target":   torch.zeros(B, la),
            "encoder_cat":      torch.zeros(B, lb, 1, dtype=torch.long),
            "decoder_cat":      torch.zeros(B, la, 1, dtype=torch.long),
            "encoder_lengths":  torch.full((B,), lb, dtype=torch.long),
            "decoder_lengths":  torch.full((B,), la, dtype=torch.long),
            "decoder_time_idx": torch.arange(la)[None, :].expand(B, -1).contiguous(),
            "groups":           torch.zeros(B, 1, dtype=torch.long),
            "target_scale":     torch.stack([torch.zeros(B), torch.ones(B)], dim=1),
        }

    # ──────────────────────────────────────────────────────────
    #  Interface compatibility
    # ──────────────────────────────────────────────────────────
    @override
    def sample_task(self, idx):
        return self.sample_window(idx)

    @override
    def split_data(self, data):
        pass

    @classmethod
    def load_valid_plants(cls, txt_path: str) -> set:
        """Read the offline validity whitelist (plants >= PCT_KEEP)."""
        if not os.path.exists(txt_path):
            print(f"[PVTimeSeriesDatasetV2] ⚠️  valid_plants.txt not found — "
                  f"using all plants")
            return set()
        with open(txt_path) as fh:
            valid = {line.strip() for line in fh if line.strip()}
        print(f"[PVTimeSeriesDatasetV2] ✅ Loaded {len(valid)} valid plants "
              f"from whitelist")
        return valid

    def _set_timestamps(
        self, plant_id: str,
        start_row: int, n_support: int,
        n_query: int,  query_start_row: int,
    ) -> None:
        """Compute look-ahead timestamps for query and support windows."""
        ts = self.__class__._timestamps.get(plant_id)
        if ts is None:
            self._last_timestamps = None
            self.support_timestamps = None
            return

        lb, la, stride = self.look_back_steps, self.look_ahead_steps, self.window_stride

        def _gather(base: int, n: int) -> np.ndarray:
            if n == 0:
                return np.empty((0, la), dtype=ts.dtype)
            idx = (base + lb
                + np.arange(n)[:, None] * stride   # [n, 1]
                + np.arange(la)[None, :])           # [1, la]  → [n, la]
            return ts[np.clip(idx, 0, len(ts) - 1)]

        self._last_timestamps   = _gather(query_start_row, n_query)   # [n_q, la]
        self.support_timestamps = _gather(start_row,       n_support) # [n_s, la]

    def _target_span(self, table: pa.Table, start_row: int, n_windows: int) -> np.ndarray:
        """Raw target values covering the full row-span of a support/query block."""
        span_rows = (n_windows - 1) * self.window_stride + self.window_steps
        chunk = table.slice(start_row, span_rows).column(self.TARGET_COL)
        return chunk.to_numpy(zero_copy_only=False)

    def _day_fraction_valid(self, y: np.ndarray) -> float:
        """Fraction of full 96-step days with >= MIN_NNZ_DAY nonzero steps."""
        n_days = len(y) // self.DAY_STEPS
        if n_days == 0:
            return 0.0
        d = y[: n_days * self.DAY_STEPS].reshape(n_days, self.DAY_STEPS)
        nnz_per_day = (d != 0).sum(axis=1)
        return float((nnz_per_day >= self.MIN_NNZ_DAY).mean())

    def sample_window(self, idx) -> Tuple:
        """Samples one task for the given pool index.
        - Val/test: deterministic, unfiltered sampling via `_sample_once`
          (seeded by `_eval_seed + idx` for reproducibility across epochs).
        - Train + random_sampling: unfiltered random plant via `_sample_once`.
        - Train + contiguous sampling (default): rejection-sampling loop that
          retries different start offsets (and, after `MAX_RESAMPLE` failures,
          switches plant) until both the support and query target spans meet
          the day-validity thresholds (`THETA_SUPPORT`/`THETA_QUERY`), to avoid
          training on trivial all-zero-power windows.
        """
        # ── EVAL / VAL: unchanged, deterministic, no filtering ──
        if self._is_val or self._is_test:
            return self._sample_once(self.pool[idx], idx)

        # ── RANDOM sampling path: unchanged ──
        if self.random_sampling:
            return self._sample_once(random.choice(self.pool), idx)

        # ── TRAIN, contiguous path: day-validity filter ──
        task_name       = f"task_{idx:05d}"
        total_attempts  = 0                      # over all plants
        plants_tried    = []                     # (plant_id, n_attempts, best_frac_s, best_frac_q)
        min_needed = self._compute_min_windows()
        
        for _switch in range(self.MAX_PLANT_SWITCH):
            plant_id = random.choice(self.pool)
            table    = self._arrow_cache[plant_id]
            n_rows   = table.num_rows

            n_windows = (n_rows - self.window_steps) // self.window_stride + 1
            # MAX_WINDOWS = max(4, self.gap//self.window_stride + 2)
            if n_windows < min_needed: # if n_windows < max_win: # if n_windows <= 0
                self.n_rejected_plants += 1
                plants_tried.append((plant_id, 0, 0.0, 0.0))
                continue

            n_support, n_query, gap, total_needed = self._compute_split(n_windows)
            max_start = n_windows - total_needed

            plant_attempts = 0
            best_fs, best_fq = 0.0, 0.0

            for _attempt in range(self.MAX_RESAMPLE + 1):     # 0..5 → 6 tries
                total_attempts += 1
                plant_attempts += 1

                start_idx       = random.randint(0, max(0, max_start))
                start_row       = start_idx * self.window_stride
                query_start_row = start_row + (n_support * self.window_stride) + gap

                y_s = self._target_span(table, start_row,       n_support)
                y_q = self._target_span(table, query_start_row, n_query)

                frac_s = self._day_fraction_valid(y_s)
                frac_q = self._day_fraction_valid(y_q)
                best_fs, best_fq = max(best_fs, frac_s), max(best_fq, frac_q)

                if (frac_s >= self.THETA_SUPPORT and frac_q >= self.THETA_QUERY):
                    self._last_plant_id       = plant_id
                    self._last_rated_power_kW = self._rated_power.get(plant_id)

                    print(
                        f"[{task_name}] ✅ ACCEPT | plant={plant_id} | "
                        f"plant_switches={_switch} | attempts_total={total_attempts} | "
                        f"resamples_this_plant={_attempt} | "
                        f"frac_support={frac_s:.2f} (θ={self.THETA_SUPPORT}) | "
                        f"frac_query={frac_q:.2f} (θ={self.THETA_QUERY}) | "
                        f"n_s={n_support}, n_q={n_query}, gap={gap} | "
                        f"start_row={start_row}"
                    )
                    return self._build_task(table, plant_id, start_row,
                                            n_support, n_query, query_start_row)

                self.n_rejected_windows += 1

            # 5 resamples failed → switch plant
            self.n_rejected_plants += 1
            plants_tried.append((plant_id, plant_attempts, best_fs, best_fq))
            print(
                f"[{task_name}] ↻ plant rejected | plant={plant_id} | "
                f"attempts={plant_attempts} | best_frac_s={best_fs:.2f} | "
                f"best_frac_q={best_fq:.2f} → switching plant "
                f"({_switch + 1}/{self.MAX_PLANT_SWITCH})"
            )

        # ── Final failure: dump full diagnostics ──
        print(f"[{task_name}] ❌ FAILED after {self.MAX_PLANT_SWITCH} plant switches "
            f"| total_attempts={total_attempts}")
        for pid, att, fs, fq in plants_tried:
            print(f"    plant={pid} | attempts={att} | best_frac_s={fs:.2f} | best_frac_q={fq:.2f}")

        raise RuntimeError(
            f"No valid task after {self.MAX_PLANT_SWITCH} plant switches "
            f"(idx={idx}); lower min_nnz_day or theta.")


    def _compute_split(self, n_windows):
        """Your existing gap-aware split math, unchanged."""
        max_win = (self.max_windows_train if state.train
                else self.max_windows_val if self._is_val
                else self.max_windows_test)
        if max_win is None:
            max_win = self.max_windows

        batch_size = min(max_win, n_windows) if max_win else n_windows
        gap = self.gap
        usable = batch_size - gap
        if usable < 4:
            gap, usable = 0, batch_size

        n_support = max(1, int(usable * self.train_ratio))
        n_query   = max(1, usable - n_support)
        total_needed = n_support + gap + n_query
        if total_needed > n_windows:
            raise ValueError(  # ← wird schon in sample_window abgefangen
                f"Not enough windows: need {total_needed}, have {n_windows}")

        return n_support, n_query, gap, total_needed

    def full_task(self, idx) -> Tuple:
        """ Returns all windows from one plant without support and query splits
        TFT:  X = (X_enc [N, lb, F_enc], X_dec [N, la, F_dec]),  y [N, la, 1]
        LSTM: X = [N, lb, F],                                      y [N, la, 1]
        """
        plant_id = self.pool[idx]
        table    = self._arrow_cache[plant_id]
        n_rows   = table.num_rows

        n_windows_total = (n_rows - self.window_steps) // self.window_stride + 1
        if n_windows_total <= 0:
            raise ValueError(
                f"Plant {plant_id}: {n_rows} rows — at least "
                f"{self.window_steps} needed."
            )

        max_win = self.max_windows_train
        if max_win and n_windows_total > max_win:
            max_start = n_windows_total - max_win
            start_idx = random.randint(0, max_start)   # ← fixes temporal bias
            start_row = start_idx * self.window_stride
            n_windows = max_win
        else:
            start_row, n_windows = 0, n_windows_total

        windows = self._extract_windows(table, start_row, n_windows)   # [N, W, C]

        lb         = self.look_back_steps
        feat_idx   = self._feat_idx
        target_idx = self._target_idx

        if self.use_tft or self.use_pf_tft:
            dec_idx = self._dec_idx
            X_enc = windows[:, :lb, :][:, :, feat_idx]        # [N, lb, F_enc]
            X_dec = windows[:, lb:, :][:, :, dec_idx]         # [N, la, F_dec]
            y     = windows[:, lb:, :][:, :, [target_idx]]    # [N, la,  1  ]
            X_enc, X_dec, y = self._filter_nan(X_enc, X_dec, y)
            return (
                (
                    torch.from_numpy(X_enc).clone(),
                    torch.from_numpy(X_dec).clone(),
                ),
                torch.from_numpy(y).clone(),
            )
        else:
            X = windows[:, :lb, :][:, :, feat_idx]            # [N, lb, F]
            y = windows[:, lb:, :][:, :, [target_idx]]        # [N, la,  1]
            X, y = self._filter_nan(X, y)
            return torch.from_numpy(X).clone(), torch.from_numpy(y).clone()

    def _compute_min_windows(self) -> int:
        """Absolute minimum windows for a valid support + gap + query split."""
        return self.gap + 4   # 2 support + gap + 2 query