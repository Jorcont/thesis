import os
from typing import List, Tuple
from typing_extensions import override

from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler
from data.dataset_classes.PVTimeSeriesDataset import PVTimeSeriesDataset
from conf.data_configs import DataConfig
from data.dataset_classes.PVPreprocessor import PVPreprocessor

class PVTaskSampler(SupervisedLearningTaskSampler):
    """
    Creates a pool with the different .csv files (tasks)
    Looks in the directory data/PV_data/ for them
    """

    FEATURES_PREFIX = "pattern_"
    TARGET_PREFIX   = "target_"
    FILE_EXTENSION  = ".csv"

    def __init__(self, config: dict):
        # Inject dataset_class automatically — no need to pass it manually
        config["dataset_class"] = PVTimeSeriesDataset
        super().__init__(config)

    @override
    def create_task_pool(self):
        """Scan data directory for (features, target) CSV pairs."""
        data_dir = DataConfig.get_config().config.get(
            "data_path", "data/PV_data/"
        ) if DataConfig.get_config() else "data/PV_data/"

        files = os.listdir(data_dir)

        self.task_pool: List[Tuple[str, str]] = []
        for f in files:
            if f.startswith(self.FEATURES_PREFIX) and f.endswith(self.FILE_EXTENSION):
                plant_id    = f[len(self.FEATURES_PREFIX):-len(self.FILE_EXTENSION)]
                target_file = f"{self.TARGET_PREFIX}{plant_id}{self.FILE_EXTENSION}"

                if target_file in files:
                    self.task_pool.append((
                        os.path.join(data_dir, f),
                        os.path.join(data_dir, target_file),
                    ))
                else:
                    print(f"[PVTaskSampler] ⚠️  Skipping '{plant_id}': no target file.")

        print(
            f"[PVTaskSampler] Found {len(self.task_pool)} plants:\n"
            f"  {[os.path.basename(p[0]) for p in self.task_pool]}"
        )

    @override
    def create_datasets(self):
        self.train_dataset = PVTimeSeriesDataset(pool=self.param_ranges["train"])
        self.val_dataset = PVTimeSeriesDataset(pool=self.param_ranges["val"])
        self.test_dataset = PVTimeSeriesDataset(pool=self.param_ranges["test"])
        self._prewarm_cache()  # ← add

    def _prewarm_cache(self):
        """Preload all plants into cache before training starts."""
        import time
        all_paths = (
                self.param_ranges["train"] +
                self.param_ranges["val"] +
                self.param_ranges["test"]
        )
        print(f"[PVTaskSampler] Pre-warming cache for {len(all_paths)} plants...")
        for features_path, target_path in all_paths:
            cache_key = (features_path, target_path)
            if cache_key not in PVTimeSeriesDataset._cache:
                t0 = time.perf_counter()
                preprocessor = PVPreprocessor(
                    features_csv_path=features_path,
                    target_csv_path=target_path,
                    interpolate=self.train_dataset.interpolate,
                    align_time=self.train_dataset.align_time,
                )
                PVTimeSeriesDataset._cache[cache_key] = (
                    preprocessor.get_dataframe(),
                    preprocessor,
                )
                print(f"  cached {os.path.basename(features_path)} "
                      f"in {(time.perf_counter() - t0) * 1e3:.0f} ms")
        print(f"[PVTaskSampler] Cache ready — {len(PVTimeSeriesDataset._cache)} plants loaded")