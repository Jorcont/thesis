''' Task-Sampler to create and sample from Prototype Learning task pool.'''

from typing_extensions import override
from typing import List, Tuple, Dict
import os
from datasets import load_from_disk
import random
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler
from conf.data_configs import DataConfig

class PrototypeLearningTaskSampler(SupervisedLearningTaskSampler):
    @override
    def create_task_pool(self):
        # Todo: Jetzt ist es ja doch wieder doppelt verschachtelt.
        self.data_config = DataConfig.get_config().config
        self.data_path:str = self.data_config.data_path
        data: DatasetDict = load_dataset(str(self.data_path))
        if not self.data_config.cifar:
            data = self.add_label_column(data)

        # Concatenate every Dataset in the DatasetDict to create a single task pool
        self.task_pool = concatenate_datasets(list(data.values()))
        # Set column names.
        self.label_name = self.data_config.label_name

    def add_label_column(self, data: DatasetDict) -> DatasetDict:
        all_class_names = set()
        for split in data.keys():
            class_names = [key.split("/")[1] for key in data[split]["__key__"]]
            all_class_names.update(class_names)

        # Create mapping: string → integer
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(all_class_names))}

        # Add INTEGER label column
        for split in data.keys():
            labels = [self.class_to_idx[key.split("/")[1]] for key in data[split]["__key__"]]
            data[split] = data[split].add_column(self.data_config.label_name, labels)

        return data

    @override
    def determine_len(self, data: DatasetDict):
        ''' The length of image datsets is defined as the number of classes.'''
        return len(set(data[self.label_name]))

    @override
    def split_pool(self):
        """ Splits the task pool into training, validation and testing tasks,
        and creates the respective torch datasets."""
        self.param_ranges: Dict[str, Dataset] = {}

        for key in self.splits.keys():
            self.determine_split_idx(key)
            all_classes = list(range(self.total_length))
            random.shuffle(all_classes)

            self.param_ranges['train'] = self.task_pool.filter(lambda x: x[self.label_name] in all_classes[:self.split_idx[0]])
            self.param_ranges['val'] = self.task_pool.filter(lambda x: x[self.label_name] in all_classes[self.split_idx[0]:self.split_idx[1]])
            self.param_ranges['test'] = self.task_pool.filter(lambda x: x[self.label_name] in all_classes[self.split_idx[1]:])
