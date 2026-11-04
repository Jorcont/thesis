''' Configs for data generators like physical models.'''

from dataclasses import dataclass, field
from typing import List, Literal

from conf.ConfigBase import ConfigBase

@dataclass
class GeneratorConfig(ConfigBase):
    name: Literal['Dummmy'] = 'Dummy'
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == 'Dummy':
            from data.data_generators.physical_models.DummyModel import DummyModel
            self.config = DummyConfig(**self.config).__dict__
            self.generator = DummyModel(self.config)
        else:
            raise ValueError("There exists no config for this generator name. Make sure that a sub-generator-config with this name exists.")

@dataclass
class DummyConfig:
    ''' Config for the dummy physical building model.'''
    length: int = 24 # number of time steps per data generation.
