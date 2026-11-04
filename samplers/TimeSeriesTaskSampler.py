''' Timeseries Task-Sampler to create and sample from tasks based on timeseries data.'''

from typing import Any, Dict, List, Tuple

from typing_extensions import override
import torch

from data.dataset_classes.TimeSeriesDataset import TimeSeriesDataset
from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler

class TimeSeriesTaskSampler(SupervisedLearningTaskSampler):
    ''' Supervised learning class with timeseries input It receives a data config and
    forwards it to the sampler's dataset class that, in this case, returns pytorch_forecasting TimeSeriesDatasets.
    These datasets are used for data spliting, model creation and trainer configuration.
    '''
    @override
    def create_task_pool(self):
        """ Creates the task pool from which training, validation and test tasks are sampled.
        In this case, it is the set of possible generator parameters.
        This set will be used to sample from in order to create new tasks.
        Each parameter is given with an interval of its min and max values.
        """
        # Join the paths --> Read from YAML, so
        self.task_pool: Dict[str, List[float]] = {"primary_supply_temp": [60,120],
                      "heat_exchanger_transfer_area": [0.5,50.0],
                      "max_primary_mass_flow": [0.01,20.0],
                      "building_thermal_resistance": [1e-15,0.02],
                      "building_thermal_capacity": [1e6,1e10],
                      "indoor_temp_setpoint": [18.0, 25.0],}

    @override
    def determine_split_idx(self, key):
        ''' Helper function for data spliting that returns split indices.
        This version splits intervals.
        '''
        upper_bound = self.task_pool[key][1] # 120
        lower_bound = self.task_pool[key][0] # 60
        interval_len = upper_bound - lower_bound # 60
        # Calculate edge values for the 3er-split:
        lower_value = lower_bound + interval_len * self.splits[key][0]
        upper_value = lower_value + interval_len * self.splits[key][1]
        self.split_idx:List[float] = [ float(lower_value), float(upper_value)]
        if key== "building_thermal_resistance":
            print( lower_bound,interval_len, upper_bound, lower_value, upper_value)

    @override
    def split_pool(self):
        """ Splits the task pool into training, validation and testing tasks,
        and creates the respective torch datasets."""
        for key in self.splits.keys():
            self.determine_split_idx(key)

            self.param_ranges['train'][key] = [self.task_pool[key][0], self.split_idx[0]]
            self.param_ranges['val'][key] = [self.split_idx[0], self.split_idx[1]]
            self.param_ranges['test'][key] = [self.split_idx[1], self.task_pool[key][1]]
