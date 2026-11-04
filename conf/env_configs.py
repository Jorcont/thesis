''' General environment configs.'''

from dataclasses import dataclass, field
from typing import List, Literal
import gymnasium as gym

from conf.ConfigBase import ConfigBase

@dataclass
class EnvConfig(ConfigBase):
    ''' Orchestrates environment config creation.
    Every sub-env-config called from this class must have a make_env function
    determining how to create the respective environment with that env_config.
    '''
    name: Literal['CartPole'] = 'CartPole'
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == 'CartPole':
            self.config = CartPoleConfig(self.config).__dict__
        else:
            raise ValueError("There exists no config for this environment name. Make sure that a sub-env-config with this name exists.")

@dataclass
class CartPoleConfig:
    ''' Dummy example for a specific environment config.'''
    masscart: float= 1.0
    masspole: float =0.1
    length: float = 0.5
    force_mag: float = 10.0

    def make_env(self, env_config):
        ''' Defines how to create environments.
        This function must neccessarily be re-written for each use case.
        Here, we use cartpole as an example.
        Note that the dummy values of this config do not influence env init.
        But this is not the case for more complex envs.
        '''
        return gym.make('CartPole-v1')
