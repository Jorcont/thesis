from dataclasses import dataclass, field
from importlib import import_module
from typing import Literal, Dict
import torch.nn as nn

from conf.ConfigBase import ConfigBase

@dataclass
class TaskConfig(ConfigBase):
    """ Base class for task configurations.

        Attributes:
            name (optional): Sets the task type.
            config (optional): A dictionary of arguments to be passed to the task constructor.
    """
    name: Literal['SupervisedLearning', 'HPRegression', 'Grid', 'PrototypeLearning'] = 'SupervisedLearning'
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == 'SupervisedLearning':
            self.config = SupervisedLearningTaskConfig(**self.config).__dict__
        elif self.name == "PrototypeLearning":
            self.config = SupervisedLearningTaskConfig(**self.config).__dict__
        elif self.name == 'HPRegression':
            # Todo: Maybe the HPRegression task requires for its own task config. Keep that in mind.
            self.config = RLTaskConfig(**self.config).__dict__
        elif self.name == 'Grid':
            self.config = GridTaskConfig(**self.config).__dict__
        elif self.name == "TimeSeries":
            self.config = TimeSeriesTaskConfig(**self.config).__dict__
        else:
            raise ValueError(f"Unsupported task name: {self.name}")

        task_class_name = f"{self.name}Task"
        module_name = f"tasks.{task_class_name}"
        task_module = import_module(module_name)
        self.task_class = getattr(task_module, task_class_name)
        self.task_class.config = self.config

@dataclass
class SupervisedLearningTaskConfig:
    inner_lr: float | Dict[str, float] = 0.01
    loss_fn: nn.Module = None
    train_ratio: float = 0.5
    adaptation_steps: int = 5
    device: Literal["cpu", "cuda"] = 'cpu'

@dataclass
class RLTaskConfig(SupervisedLearningTaskConfig):
    gamma: int = 1

@dataclass
class GridTaskConfig(SupervisedLearningTaskConfig):
    scoop: int | None = None
    type_pf: Literal['ac', 'dc', 'all'] = 'ac'
    add_nminus1_data: bool = False
    data_split: float = 0.2

@dataclass
class TimeSeriesTaskConfig(SupervisedLearningTaskConfig):
    pass