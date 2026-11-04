"""
PV Time Series Dataset for Meta-Learning.

1 Task = 1 PV plant
    support set : first K windows → model adapts (few-shot)
    query set   : remaining windows → model is evaluated

Pool format (built by PVTaskSampler):
    [
        ("data/PV_data/pattern_PlantA.csv", "data/PV_data/target_PlantA.csv"),
        ...
    ]
Modes
-----
LSTM (default — no TimeSeriesDataConfig keys in YAML):
    __getitem__ → (X_support, y_support, X_query, y_query)

TFT (time_varying_known_reals):
    __getitem__ → (X_enc_s, X_dec_s), y_s, (X_enc_q, X_dec_q), y_q
    4-value interface preserved → MAML pipeline unchanged.

Sampling modes
--------------
Calendar mode (support_hours set):
    Contiguous slice [start : start + support_hours + query_hours] picked randomly.
    Support windows built from first support_hours.
    Query  windows built from next  query_hours.
    Split baked in by calendar boundary — split_data() bypassed.

Legacy mode (support_hours not set):
    max_windows cap + random offset → all windows → split by train_ratio.
"""
import random
import torch
from typing import Any, Tuple, List
from typing_extensions import override

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from data.dataset_classes.PVPreprocessor import PVPreprocessor

class PVTimeSeriesDataset(SupervisedLearningDataset):
    """
    Subclass of SupervisedLearningDataset for PV plant forecasting.

    Each __getitem__ call:
        1. Randomly picks one PV plant from pool
        2. Samples a calendar window (or legacy max_windows slice)
        3. Builds support and query windows
        4. Returns (X_support, y_support, X_query, y_query)
           where X is either a Tensor (LSTM) or a tuple of Tensors (TFT)
    """

    # ── Class-level cache (shared across all instances) ───────────
    _cache: dict = {}

    # ──────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────

    @override
    def init_data(self):
        """
        Read PV config from YAML via DataConfig singleton.

        Key new params (supervisor's feedback):
            support_hours: K — calendar hours for support set (adaptation budget)
            query_hours: calendar hours for query set (can be > support_hours)
            num_episodes: episodes per plant per epoch (was hardcoded as 30)

        Detects operating mode:
            LSTM mode — default, no TimeSeriesDataConfig keys needed
            TFT  mode — activated when 'time_varying_known_reals' is
                        present in data_config.config (YAML)
        """
        from conf.data_configs import DataConfig
        data_cfg = DataConfig.get_config()

        if data_cfg is not None:
            cfg              = data_cfg.config   # full dict from YAML
            self.train_ratio = getattr(data_cfg, "train_ratio", 0.7)
        else:
            cfg              = {}
            self.train_ratio = 0.7

        # ── Existing params (unchanged — LSTM still works) ────────
        self.look_back_hours  = cfg.get("look_back_hours",  72)
        self.look_ahead_hours = cfg.get("look_ahead_hours", 24)
        self.samples_per_hour = cfg.get("samples_per_hour",  4)
        self.interpolate      = cfg.get("interpolate", True)
        self.align_time       = cfg.get("align_time",  True)
        self.max_windows      = cfg.get("max_windows", None)
        self.num_episodes     = cfg.get("num_episodes", 30) # episodes /plant

        self.look_back_steps  = self.look_back_hours  * self.samples_per_hour  # 288
        self.look_ahead_steps = self.look_ahead_hours * self.samples_per_hour  #  96
        self.window_steps     = self.look_back_steps  + self.look_ahead_steps  # 384

        # K = support_hours → directly controls adaptation data budget
        # None → legacy mode: max_windows + train_ratio split
        self.support_hours = cfg.get("support_hours", None)
        self.query_hours   = cfg.get("query_hours",   None)

        if self.support_hours is not None:
            self.support_steps = self.support_hours * self.samples_per_hour
            self.query_steps   = (
                self.query_hours or self.support_hours
            ) * self.samples_per_hour

        # ── TFT mode detection ────────────────────────────────────
        known_reals   = cfg.get("time_varying_known_reals",   None)
        unknown_reals = cfg.get("time_varying_unknown_reals", None)
        target        = cfg.get("target", PVPreprocessor.TARGET_COLUMNS)
        if isinstance(target, str):
            target = [target]

        if known_reals is not None:
            # ── TFT mode ──────────────────────────────────────────
            self.use_tft = True

            # Encoder sees the PAST window:
            # NWP: Numerical Weather Prediction
            #   known reals   (NWP forecasts + calendar) +
            #   unknown reals excluding the target       (none here)
            # Because in the past, we have ALL of these available.
            # TODO: Recall that for this task len(enc_feature_cols) ==
            #  len(known_reals)
            self.enc_feature_cols: List[str] = known_reals + [
                c for c in (unknown_reals or []) if c not in target
            ]
            # Decoder sees the FUTURE window:
            #   known reals ONLY — the only variables we have access
            #   to for future timesteps (NWP forecasts + calendar).
            #   The actual PV output is unknown → not included.
            self.dec_feature_cols: List[str] = known_reals
            self.target_cols: List[str] = target

        else:
            # ── LSTM mode (default) ───────────────────────────────
            self.use_tft          = False
            self.enc_feature_cols = None   # uses preprocessor.FEATURE_COLUMNS
            self.dec_feature_cols = None
            self.target_cols      = PVPreprocessor.TARGET_COLUMNS

        # ── Profiling counters ────────────────────────────────────────
        self._prof = {
            "cache_hits": 0,
            "cache_misses": 0,
            "t_preprocess": [],
            "t_build_windows": [],
            "t_sample_total": [],
        }

        # ── Print summary ─────────────────────────────────────────
        mode_str = "TFT" if self.use_tft else "LSTM"
        sampling_str = (
            f"calendar  K=support={self.support_hours}h  "
            f"query={self.query_hours or self.support_hours}h"
            if self.support_hours is not None
            else f"legacy  max_windows={self.max_windows}  train_ratio={self.train_ratio}"
        )

        print(
            f"[PVTimeSeriesDataset] Config:\n"
            f"  look_back={self.look_back_steps} steps  "
            f"look_ahead={self.look_ahead_steps} steps  "
            f"mode={mode_str}\n"
            f"  sampling={sampling_str}  episodes/plant={self.num_episodes}"
        )
        if self.use_tft:
            print(
                f"  enc_features ({len(self.enc_feature_cols)}): "
                f"{self.enc_feature_cols}\n"
                f"  dec_features ({len(self.dec_feature_cols)}): "
                f"{self.dec_feature_cols}"
            )

    # ──────────────────────────────────────────────────────────────
    #  Entry point
    # ──────────────────────────────────────────────────────────────

    @override
    def __getitem__(self, idx):
        """
        Route to calendar-window or legacy sampling based on config.

        Calendar mode (support_hours set):
            Returns (X_support, y_support, X_query, y_query) directly.
            Split is baked in by calendar boundary — split_data() bypassed.

        Legacy mode (support_hours not set):
        sample_task() → split_data() → (X_support, y_support, X_query, y_query)
        """
        if self.support_hours is not None:
            return self._sample_calendar_task()
        data = self.sample_task()
        return self.split_data(data)

    # ──────────────────────────────────────────────────────────────
    #  Calendar-window sampling (new)
    # ──────────────────────────────────────────────────────────────

    def _sample_calendar_task(self) -> Tuple:
        """
        Sample one PV plant and return support/query sets from a
        contiguous calendar window of support_hours + query_hours.

        Support: first support_hours → model adapts   (K shots)
        Query  : next  query_hours   → model evaluated (can be longer than support)

        Random start ensures all seasons are covered across episodes.

        LSTM returns: (X_support, y_support, X_query, y_query)
        TFT  returns: (X_enc_s, X_dec_s), y_s, (X_enc_q, X_dec_q), y_q
        """
        import time
        t_total = time.perf_counter()

        features_path, target_path = random.choice(self.pool)
        cache_key = (features_path, target_path)

        if cache_key not in PVTimeSeriesDataset._cache:
            self._prof["cache_misses"] += 1
            t0 = time.perf_counter()
            preprocessor = PVPreprocessor(
                features_csv_path=features_path,
                target_csv_path=target_path,
                interpolate=self.interpolate,
                align_time=self.align_time,
            )
            PVTimeSeriesDataset._cache[cache_key] = (
                preprocessor.get_dataframe(),
                preprocessor,
            )
            self._prof["t_preprocess"].append(time.perf_counter() - t0)
        else:
            self._prof["cache_hits"] += 1

        data, preprocessor = PVTimeSeriesDataset._cache[cache_key]

        # ── Validate enough data for support + query ──────────────
        min_rows = self.support_steps + self.query_steps
        if len(data) < min_rows:
            raise ValueError(
                f"Plant '{features_path}' has {len(data)} rows "
                f"({len(data) / self.samples_per_hour:.0f}h) — need "
                f"{min_rows} rows for support={self.support_hours}h "
                f"+ query={self.query_hours or self.support_hours}h."
            )

        # ── Validate window_steps fits within each split ──────────
        if self.support_steps < self.window_steps:
            raise ValueError(
                f"support_hours={self.support_hours}h ({self.support_steps} steps) "
                f"< window_steps={self.window_steps} steps "
                f"(look_back + look_ahead = "
                f"{self.window_steps / self.samples_per_hour:.0f}h). "
                f"Increase support_hours."
            )
        if self.query_steps < self.window_steps:
            raise ValueError(
                f"query_hours={self.query_hours}h ({self.query_steps} steps) "
                f"< window_steps={self.window_steps} steps. Increase query_hours."
            )

        # ── Random contiguous calendar window ─────────────────────
        max_start = max(0, len(data) - self.support_steps - self.query_steps)
        start = random.randint(0, max_start)

        # Slices are exactly support_steps / query_steps rows each —
        # _build_windows derives n_windows = len(slice) - window_steps + 1
        support_slice = data.iloc[
            start: start + self.support_steps
        ].reset_index(drop=True)

        query_slice = data.iloc[
            start + self.support_steps: start + self.support_steps + self.query_steps
        ].reset_index(drop=True)

        # ── Apply max_windows cap within each calendar slice ──────
        # Preserves the seasonal context set by support_hours / query_hours,
        # while keeping the inner-loop window count at ≤ max_windows.
        if self.max_windows is not None:
            support_slice = self._cap_slice(support_slice)
            query_slice = self._cap_slice(query_slice)

        # ── Build windows from each slice ─────────────────────────
        t0 = time.perf_counter()
        if self.use_tft:
            X_enc_s, X_dec_s, y_s = self._build_windows_tft(support_slice)
            X_enc_q, X_dec_q, y_q = self._build_windows_tft(query_slice)
            result = (X_enc_s, X_dec_s), y_s, (X_enc_q, X_dec_q), y_q
        else:
            X_s, y_s = self._build_windows(support_slice, preprocessor)
            X_q, y_q = self._build_windows(query_slice, preprocessor)
            result = X_s, y_s, X_q, y_q

        self._prof["t_build_windows"].append(time.perf_counter() - t0)
        self._prof["t_sample_total"].append(time.perf_counter() - t_total)

        return result

    def _cap_slice(self, slice_df):
        """
        If the slice produces more than max_windows windows, randomly
        pick a contiguous sub-slice of exactly max_windows + window_steps - 1 rows
        (the minimum rows to produce exactly max_windows windows).

        Example:
            support_hours=720 → 2880 steps → 2497 possible windows
            max_windows=45    → cap to 45 + 384 - 1 = 428 rows
            Offset is random → diverse sub-windows within the month
        """
        n_possible = len(slice_df) - self.window_steps + 1
        if n_possible > self.max_windows:
            offset = random.randint(0, n_possible - self.max_windows)
            slice_df = slice_df.iloc[
                offset: offset + self.max_windows + self.window_steps - 1
            ].reset_index(drop=True)
        return slice_df

    # ──────────────────────────────────────────────────────────────
    #  Legacy Task sampling (max_windows + train_ratio)
    # ──────────────────────────────────────────────────────────────

    @override
    def sample_task(self) -> Tuple:
        """
        Legacy sampling mode (support_hours not set).
        Apply max_windows cap + random seasonal offset,
        return all windows unsplit — split_data() applies train_ratio

        LSTM returns : (X, y)
            X : [N, look_back,  n_features]
            y : [N, look_ahead, n_targets]

        TFT returns  : (X_enc, X_dec, y)
            X_enc : [N, look_back,  n_enc_features]
            X_dec : [N, look_ahead, n_dec_features]
            y     : [N, look_ahead, n_targets]
        """
        import time
        t_total = time.perf_counter()

        features_path, target_path = random.choice(self.pool)
        cache_key = (features_path, target_path)

        if cache_key not in PVTimeSeriesDataset._cache:
            self._prof["cache_misses"] += 1
            t0 = time.perf_counter()
            preprocessor = PVPreprocessor(
                features_csv_path = features_path,
                target_csv_path   = target_path,
                interpolate       = self.interpolate,
                align_time        = self.align_time,
            )
            PVTimeSeriesDataset._cache[cache_key] = (
                preprocessor.get_dataframe(),
                preprocessor,
            )
            self._prof["t_preprocess"].append(time.perf_counter() - t0)
        else:
            self._prof["cache_hits"] += 1

        data, preprocessor = PVTimeSeriesDataset._cache[cache_key]

        # ── Apply max_windows cap with random seasonal offset ─────
        n_total = len(data) - self.window_steps + 1
        if n_total <= 0:
            raise ValueError(
                f"Not enough data: {len(data)} rows, "
                f"need at least {self.window_steps}."
            )
        if self.max_windows is not None and n_total > self.max_windows:
            start     = random.randint(0, n_total - self.max_windows)
            n_windows = self.max_windows
        else:
            start     = 0
            n_windows = n_total

        # Slice: n_windows windows need n_windows + window_steps - 1 rows
        data_slice = data.iloc[
            start : start + n_windows + self.window_steps - 1
        ].reset_index(drop=True)


        t0 = time.perf_counter()
        # Identify which windows should we return
        if self.use_tft:
            result = self._build_windows_tft(data_slice)
        else:
            result =  self._build_windows(data_slice, preprocessor)

        self._prof["t_build_windows"].append(time.perf_counter() - t0)
        self._prof["t_sample_total"].append(time.perf_counter() - t_total)

        return result
    # ──────────────────────────────────────────────────────────────
    #  LSTM window builder
    # ──────────────────────────────────────────────────────────────

    def _build_windows(
        self,
        data_slice,
        preprocessor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build all sliding windows from a pre-sliced DataFrame.
        n_windows = len(data_slice) - window_steps + 1"""
        n_windows = len(data_slice) - self.window_steps + 1
        if n_windows <= 0:
            raise ValueError(
                f"Slice too short: {len(data_slice)} rows, "
                f"need at least {self.window_steps}."
            )

        X_list, y_list = [], []
        for i in range(n_windows): #start, start + n_windows):
            lb_end = i + self.look_back_steps
            la_end = lb_end + self.look_ahead_steps

            x = torch.FloatTensor(
                data_slice.iloc[i:lb_end][preprocessor.FEATURE_COLUMNS].values
            )
            y = torch.FloatTensor(
                data_slice.iloc[lb_end:la_end][preprocessor.TARGET_COLUMNS].values
            )
            X_list.append(x)
            y_list.append(y)

        return torch.stack(X_list), torch.stack(y_list)  # [N, steps, F/T]

    # ──────────────────────────────────────────────────────────────
    #  TFT window builder
    # ──────────────────────────────────────────────────────────────

    def _build_windows_tft(
        self,
        data_slice,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sliding windows for TFT.

        Encoder window  (past look_back steps):
            time_varying_known_reals   → NWP forecasts + calendar
            time_varying_unknown_reals → actual measurements (excl. target)

        Decoder window  (future look_ahead steps):
            time_varying_known_reals ONLY
            (the only variables known for future timesteps)

        Target window (future look_ahead steps):
            target column — actual PV output, never known in advance
        """
        n_windows = len(data_slice) - self.window_steps + 1
        if n_windows <= 0:
            raise ValueError(
                f"Slice too short: {len(data_slice)} rows, "
                f"need at least {self.window_steps}."
            )

        # Column validation, before loading all the data
        missing_enc = [
            c for c in self.enc_feature_cols if c not in data_slice.columns]
        missing_dec = [
            c for c in self.dec_feature_cols if c not in data_slice.columns]
        missing_tgt = [
            c for c in self.target_cols      if c not in data_slice.columns]
        if any([missing_enc, missing_dec, missing_tgt]):
            raise KeyError(
                f"Missing columns in DataFrame:\n"
                f"  encoder : {missing_enc}\n"
                f"  decoder : {missing_dec}\n"
                f"  target  : {missing_tgt}\n"
                f"Available : {data_slice.columns.tolist()}"
            )

        enc_list, dec_list, y_list = [], [], []
        for i in range(n_windows): #start, start + n_windows):
            # i -> n_windows.
            lb_end = i + self.look_back_steps
            la_end = lb_end + self.look_ahead_steps

            # Past window — all features available historically
            x_enc = torch.FloatTensor(
                data_slice.iloc[i:lb_end][self.enc_feature_cols].values
            )                                            # [look_back,  n_enc]

            # Future window — only what is known in advance
            x_dec = torch.FloatTensor(
                data_slice.iloc[lb_end:la_end][self.dec_feature_cols].values
            )                                               # [look_ahead, n_dec]

            # Target — actual future PV output
            y = torch.FloatTensor(
                data_slice.iloc[lb_end:la_end][self.target_cols].values
            )                                               # [look_ahead, n_tgt]

            enc_list.append(x_enc)
            dec_list.append(x_dec)
            y_list.append(y)

        return (
            torch.stack(enc_list),   # [N, look_back,  n_enc]
            torch.stack(dec_list),   # [N, look_ahead, n_dec]
            torch.stack(y_list),     # [N, look_ahead, n_targets]
        )

    # ──────────────────────────────────────────────────────────────
    #  split_data — legacy path only
    # ──────────────────────────────────────────────────────────────

    @override
    def split_data(
        self,
        data: Tuple,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Split windows into support (adapt) and query (evaluate) sets.

        LSTM  input  : (X, y)
              returns: X_support, y_support, X_query, y_query
                       all Tensors

        TFT   input  : (X_enc, X_dec, y)
              returns: (X_enc_s, X_dec_s), y_s, (X_enc_q, X_dec_q), y_q
                       X is a tuple of Tensors, y is a Tensor

        The 4-value interface is kept in both cases so the
        MAML pipeline does not need modification.
        """
        if self.use_tft:
            X_enc, X_dec, y = data
            # Integer for slicing. X_enc_shape[0] --> n_windows
            split     = int(X_enc.shape[0] * self.train_ratio)

            return (
                (X_enc[:split], X_dec[:split]),   # X_support — tuple
                y[:split],                         # y_support
                (X_enc[split:], X_dec[split:]),   # X_query   — tuple
                y[split:],                         # y_query
            )

        else:
            # ── LSTM logic ───────────────────
            X, y      = data
            n_windows = X.shape[0]
            # Integer for slicing
            split     = int(n_windows * self.train_ratio)
            return X[:split], y[:split], X[split:], y[split:]


    @override
    def __len__(self):
        """
        Number of tasks per epoch.
        Each call to __getitem__ samples a new random plant.
        """
        # num_episodes = 30
        print("Important number to consider the computation time")
        print(f"Training tasks: {len(self.pool)}\n")
        print(f"{len(self.pool) * self.num_episodes} tasks / epoch\n")
        print("Iterations per epoch increase accordingly, "
              "since they depend on the batch_size")
        return len(self.pool) * self.num_episodes   # X episodes per plant per epoch

    def print_profiling(self):
        import numpy as np
        p = self._prof

        def fmt(lst):
            if not lst:
                return "N/A (no samples)"
            return f"{np.mean(lst)*1e3:.1f} ms / {np.max(lst)*1e3:.1f} ms"

        print(
            f"\n[PVTimeSeriesDataset] Profiling summary:\n"
            f"  cache hits/misses     : {p['cache_hits']} / {p['cache_misses']}\n"
            f"  preprocess     avg/max: {fmt(p['t_preprocess'])}\n"
            f"  build_windows  avg/max: {fmt(p['t_build_windows'])}\n"
            f"  sample_total   avg/max: {fmt(p['t_sample_total'])}"
        )