''' General Task Sampler configs.'''

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Union

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from conf.ConfigBase import ConfigBase

@dataclass
class SamplerConfig(ConfigBase):
    """ Base class for task sampler configurations.
    Determines, how the task pool creation and the split into meta-train/val/test tasks is handled.
    For each use case, a new sampler class and hence sampler config is expected.
    Those must be created consistent with the corresponding Dataset classes handling the ask-level data collection/sampling.
    In order to create an new sampler:
    1. Create a sub-config class like those below. You are encoraged to inherit from one of he already existing sub-configs.
    2. Add your sampler name to the possible names of this base class.
    3. Write the corresponding if statement in the post_init of this base class.
    Note that sampler and dataset classes might be merged in the future.

        Attributes:
            name (optional): Sets the task type.
            config (optional): A dictionary of arguments to be passed to the sampler constructor.
    """
    name: Literal['PVTimeSeries',
    'SupervisedLearning',
    'TimeSeries', 'Grid', 'PrototypeLearning'] = 'SupervisedLearning'
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "SupervisedLearning":
            from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            self.sampler = SupervisedLearningTaskSampler(self.sampler_config.__dict__)
        elif self.name == "TimeSeries":
            from samplers.TimeSeriesTaskSampler import TimeSeriesTaskSampler
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            self.sampler = TimeSeriesTaskSampler(self.sampler_config.__dict__)
        elif self.name == "Grid":
            from samplers.GridTaskSampler import GridTaskSampler
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            for key, value in self.sampler_config.__dict__.items():
                print(f"{key}: {value}")
            self.sampler = GridTaskSampler(self.sampler_config.__dict__)
        elif self.name == "PrototypeLearning":
            from samplers.PrototypeLearningTaskSampler import PrototypeLearningTaskSampler
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            self.sampler = PrototypeLearningTaskSampler(self.sampler_config.__dict__)
        elif self.name == "PVTimeSeries":
            from samplers.PVTaskSampler import PVTaskSampler
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            self.sampler = PVTaskSampler(self.sampler_config.__dict__)
        elif self.name == "PVTimeSeriesV2":
            from samplers.PVTaskSamplerV2 import PVTaskSamplerV2
            self.sampler_config = SupervisedLearningSamplerConfig(**self.config)
            self.sampler = PVTaskSamplerV2(self.sampler_config.__dict__)
        else:
            raise ValueError(f"Unknown sampler name: {self.name}")

@dataclass
class SupervisedLearningSamplerConfig:
    """
        Configuration for task sampling in meta-learning.

        Attributes:
            N_train (int): Number of training tasks per meta-training iteration.
            N_val: Total number of meta-validation tasks.
            N_test: Total number of meta-testing tasks.
            num_workers: Number of workers used for data_loading. Defaults to zero, i.e. no explicit usage.
                This feature is only relevant if lightning is used for data loading.
            splits (list): List of three floats representing the train/val/test split ratios. Must sum to 1.
            task_config: The task that will be sampled within this sampler.
                Gets set automatically by config.py if one provides a task_config in the yaml file.
            data_config: Defines the underlying datasets. Analog to task_config, set by config.py
    """
    pin_memory: bool # Gets set automatically
    dataset_class: type(SupervisedLearningDataset)
    N_train: int = 100
    N_val: int = 10 # Number of tasks for meta-validation. -1 means to use all available tasks.
    N_test: int = 10 # Analogous for testing
    num_workers: int = 0
    splits: Union[List[float], Dict[str, List[float]]] = field(default_factory=lambda: [0.9, 0.05, 0.05])
    windows_per_step: int = 512      # NEW: effective batch size of the baseline (windows per step)
    use_query_windows: bool = True   # NEW: pool support+query windows (fair data budget)
    
    def __post_init__(self):
        if isinstance(self.splits, list):
            if len(self.splits) != 3 or not all(isinstance(x, (int, float)) for x in self.splits):
                raise ValueError("splits must be a list of three numbers.")
            if not abs(sum(self.splits) - 1.0) < 1e-6:
                raise ValueError("splits must sum to 1.")
            if any(x < 0 for x in self.splits):
                raise ValueError("splits must be non-negative.")
        else:
            for key, split in self.splits.items():
                if len(split) != 3 or not all(isinstance(x, (int, float)) for x in split):
                    raise ValueError(f"Splits for {key} must be a list of three numbers.")
                if not abs(sum(split) - 1.0) < 1e-6:
                    raise ValueError(f"Splits for {key} must sum to 1.")
                if any(x < 0 for x in split):
                    raise ValueError(f"Splits for {key} must be non-negative.")

@dataclass
class RLSamplerConfig:
    """
        Configuration for task sampling in meta-learning.

        Attributes:
            N (int): Number of tasks per meta-training iteration.
            splits (list): List of three floats representing the train/val/test split ratios. Must sum to 1.
    """
    N: int = 100 # Number of tasks per meta-training iteration
    N_val: int = 10 # Number of tasks for meta-validation. -1 means to use all available tasks.
    N_test: int = 10 # Analogous for testing
    splits: List[float] = field(default_factory=lambda: [0.9, 0.05, 0.05])
    param_ranges:  Optional[dict] = None
    val_param_ranges:  Optional[dict] = None
    test_param_ranges:  Optional[dict] = None
