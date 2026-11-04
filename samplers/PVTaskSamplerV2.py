import os
import pandas as pd
import torch
from typing_extensions import override
from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler
from data.dataset_classes.PVTimeSeriesDatasetV2 import PVTimeSeriesDatasetV2

class PVTaskSamplerV2(SupervisedLearningTaskSampler):
    """
    Task sampler backed by Arrow pre-loaded tables.
    Scans plants_clean/ → splits → pre-loads ALL plants once.
    Optionally appends trafo assets to the test pool ONLY (trafo_dir + trafo_test_only flag in the YAML)
    """

    def __init__(self, config: dict):
        """
        Binds this sampler to PVTimeSeriesDatasetV2 and delegates the 
        remaining steup to the SupervisedLearningTaskSampler"""
        config["dataset_class"] = PVTimeSeriesDatasetV2
        super().__init__(config)

    @override
    def create_task_pool(self):
        """Scans `plants_dir` for available plant parquet files, excludes
        plants listed in the NaN-audit exclusion file, and — if configured —
        registers a separate trafo-asset pool to be appended to the test
        split only (see `create_datasets`)."""
        from conf.data_configs import DataConfig
        data_cfg  = DataConfig.get_config()
        self.cfg  = data_cfg.config if data_cfg else {}

        # PV plants
        plants_dir = self.cfg.get(
            "plants_dir",
            "data/PV_data",
        )
        self.plants_dir = plants_dir
        self.task_pool  = [
            f.replace(".parquet", "")
            for f in os.listdir(plants_dir)
            if f.endswith(".parquet")
        ]
        print(f"[PVTaskSamplerV2] Found {len(self.task_pool)} plants "
              f"in {plants_dir}")
       
        # ── Filter plants with high NaN fraction in target ────────────
        # Threshold and audit CSV path are configurable via YAML
        excluded_path = self.cfg.get(
            "excluded_plants_txt",
            "output/nan_audit/excluded_plants.txt"
        )
        if os.path.exists(excluded_path):
            with open(excluded_path) as f:
                bad_plants = set(f.read().splitlines())
            before         = len(self.task_pool)
            self.task_pool = [p for p in self.task_pool if p not in bad_plants]
            print(
                f"[PVTaskSamplerV2] NaN filter: "
                f"{before} → {len(self.task_pool)} plants "
                f"({before - len(self.task_pool)} excluded)"
            )
        else:
            print(f"[PVTaskSamplerV2] No exclusion list found at "
                f"{excluded_path} — no filtering applied")

        # ── Trafo test-only pool ───────────────────────────────
        trafo_dir       = self.cfg.get("trafo_dir", None)
        trafo_test_only = self.cfg.get("trafo_test_only", False)

        self.trafo_ids  = []
        self.trafo_dir  = None

        if trafo_dir and trafo_test_only:
            self.trafo_dir = trafo_dir
            self.trafo_ids = [
                f.replace(".parquet", "")
                for f in os.listdir(trafo_dir)
                if f.endswith(".parquet")
            ]
            print(f"[PVTaskSamplerV2] Trafo test-only pool: "
                  f"{len(self.trafo_ids)} assets in {trafo_dir}")
        elif trafo_dir and not trafo_test_only:
            print(f"[PVTaskSamplerV2] ⚠️  trafo_dir set but trafo_test_only=false "
                  f"— trafos NOT added. Set trafo_test_only: true to enable.")

    @override
    def create_datasets(self):
        """Split pool → pre-load ALL plants + trafo → create datasets."""

        # ── YAML-Flag steuert welche Spalten geladen werden ───────
        use_trafo = self.cfg.get("use_trafo_features", False)
        pv_cols = (
            PVTimeSeriesDatasetV2.TRAFO_FEATURE_COLS
            if use_trafo
            else PVTimeSeriesDatasetV2.FEATURE_COLS
        ) + [PVTimeSeriesDatasetV2.TARGET_COL]

        all_plants = (
            self.param_ranges["train"] +
            self.param_ranges["val"]   +
            self.param_ranges["test"]
        )

        PVTimeSeriesDatasetV2.preload(
            plant_ids  = all_plants,
            plants_dir = self.plants_dir,
            n_workers  = 32,
            columns    = pv_cols,
        )

        if self.trafo_ids and self.trafo_dir:
            trafo_cols = (
                PVTimeSeriesDatasetV2.TRAFO_FEATURE_COLS
                if use_trafo
                else PVTimeSeriesDatasetV2.FEATURE_COLS
            ) + [PVTimeSeriesDatasetV2.TARGET_COL]
            PVTimeSeriesDatasetV2.preload(
                plant_ids  = self.trafo_ids,
                plants_dir = self.trafo_dir,
                n_workers  = 32,
                columns    = trafo_cols,
            )
        # Load rate capacities (cap sum) from _renorm_report.csv
        PVTimeSeriesDatasetV2.load_rated_powers(self.plants_dir)

        mapping_path = self.cfg.get(
            "mapping_path",
            "/mnt/data/kes/data/jdomingo/nwp_parquet/plant_nwp_mapping.parquet"
        )
        PVTimeSeriesDatasetV2.load_static_features(mapping_path)

        # ── Build datasets — trafos appended to test only ─────
        test_pool = self.param_ranges["test"] + self.trafo_ids

        self.train_dataset = PVTimeSeriesDatasetV2(
            pool=self.param_ranges["train"]
        )
        self.val_dataset   = PVTimeSeriesDatasetV2(
            pool=self.param_ranges["val"]
        )
        self.test_dataset  = PVTimeSeriesDatasetV2(
            pool=test_pool#self.param_ranges["test"]
        )

        self.train_dataset._is_val = False
        self.val_dataset._is_val   = True
        self.test_dataset._is_test  = True

        print(
            f"[PVTaskSamplerV2] Datasets ready — "
            f"train: {len(self.param_ranges['train'])} | "
            f"val: {len(self.param_ranges['val'])} | "
            f"test (PV): {len(self.param_ranges['test'])} | "
            f"test (trafo): {len(self.trafo_ids)} | "
            f"test (total): {len(test_pool)} plants"
        )

    

    def make_window_collate(windows_per_step: int, use_query: bool = True):
        """Pools all windows of the sampled tasks and draws exactly
        `windows_per_step` windows -> fixed batch size, no stacking of
        variable-length tasks, bounded host RAM."""
        def collate(task_samples):
            xs, ys = [], []
            for sample in task_samples:
                X_s, y_s, X_q, y_q = sample
                parts = [(X_s, y_s)] + ([(X_q, y_q)] if use_query else [])
                for X, y in parts:
                    xs.append(X if isinstance(X, (tuple, list)) else (X,))
                    ys.append(y)

            n_feat = len(xs[0])
            X_pool = tuple(torch.cat([x[i] for x in xs], dim=0) for i in range(n_feat))
            y_pool = torch.cat(ys, dim=0)

            n = y_pool.shape[0]
            k = min(windows_per_step, n)
            idx = torch.randperm(n)[:k]
            X_batch = tuple(x[idx] for x in X_pool)
            return (X_batch if n_feat > 1 else X_batch[0]), y_pool[idx]
        return collate
    