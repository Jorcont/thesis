''' Contains data configurations that are embedded in the sampler configs.'''

from typing import Literal, Optional, Dict
from torch import device

from dataclasses import dataclass, field
from typing import List
from conf.ConfigBase import ConfigBase

@dataclass
class MetricsConfig:
    ''' Sets flags for metrics creation.'''
    name: Literal["Classification", "Regression"] = "Regression" # Back only to classification.
    device: Literal["cpu", "cuda", "cuda:0"] | device = 'cpu'
    metric_keys: List[str] = field(default_factory=lambda: ['losses'])
    generalization: bool = False
    adaptation_speed: bool = False
    sample_efficiency: bool = False
    print_results: bool = False
