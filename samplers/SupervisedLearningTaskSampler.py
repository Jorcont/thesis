''' General Task-Sampler to create and sample from task pool.'''

import os
from typing import final, Any, Dict, List, Tuple, Type
from typing_extensions import override
from importlib import import_module
from typing import final, Any, Dict, Iterator, List, Optional, Type

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.dataloader import default_collate

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from conf.data_configs import DataConfig
from algorithms.PipelineState import PipelineState as state
from data.dataset_classes.BaselineDataset import make_baseline_dataset
from conf.configs import PipelineParams
from data.window_collate import make_window_collate                      # NEW

SplitKey = Optional[str]

class SupervisedLearningTaskSampler(pl.LightningDataModule):
    ''' This is the base class for sampler creation.
    Note: In this version there is no "sampling" left in this class, as this is part of the dataset logic, now.
    Instead, the task samplers
    - Create the task pool, and
    - Divide it into meta-train/val/test pools.
    Afterwards, the corresponding dataloaders are created.
    The dataloaders are given to th pytorch_lightning trainer in LightningPipeline.py. They are created so that
    - train_dataloader samples tasks randomly.
    - val and test dataloaders examine tasks in order.
    - pin_memory attribute helps to connect to CUDA/GPU.
    - num_workers attribute determines, how many CPU workers are used to load data.
        This is for parallelization, since data loading can happen while model updates steps are still performed.
    '''
    @final
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config:Dict[str, Any] = config

        # Todo: Change N to N_train in the whole repo for better clarity.
        self.N: int = config['N_train']
        self.splits: List[float] | Dict[str, List[float]] = config['splits']
        self.num_workers: int = config['num_workers']
        self.pin_memory: bool = config['pin_memory'] # data loaded into pinned CPU memory

        self.windows_per_step: int = config.get('windows_per_step', 512)
        self.use_query_windows: bool = config.get('use_query_windows', True)
        self.data_config = DataConfig.get_config().config
        self.dataset_class: Type[SupervisedLearningDataset] = config['dataset_class']
        self.param_ranges: Dict[str, Dict[str, List[str]] | List[str]] = { "train": {}, "val": {}, "test": {}}
        # if isinstance(self.splits, list): # check if self.splits an object of class is.
        #     # Convert into dict:
        #     self.splits: Dict[str, List[float]] = {'dummy': self.splits}
        self.init_pool_and_datasets()

    def init_pool_and_datasets(self):
        '''This function gives sub-classes the possibility
        to change the order of the following function calls, see, e.g., RLTaskSampler.
        '''
        self.create_task_pool()
        self.split_pool()
        self.create_datasets()

    def create_task_pool(self):
        """Creates the task pool from which training, validation and test tasks are sampled.
        This function must potentially be changed for a specific use case.
        The task pool can be a list of indices, a dict of parameter ranges, a directory 
        with ordered sub-folders etc.
        """
        self.data_path: str = self.data_config.data_path
        self.task_pool: List[str] = [
            os.path.abspath(os.path.join(self.data_path, f))
            for f in os.listdir(self.data_path)
            if os.path.isfile(os.path.join(self.data_path, f))
        ]

    def iter_splits(self) -> Iterator[tuple[SplitKey, List[float]]]:
        """Yield split specs for one-dimensional or multi-dimensional task pools.

        ``key is None`` denotes a single, one-dimensional task pool such as a list
        of file paths or task IDs. Named keys denote one dimension of a
        multi-dimensional task pool, e.g. one sampled physical parameter, feature
        group, or interval range per key.
        """
        if isinstance(self.splits, dict):
            yield from self.splits.items()
            return

        if isinstance(self.task_pool, dict):
            for key in self.task_pool:
                yield key, self.splits
            return

        # None is the one-dimensional case; named keys above are dimensions.
        yield None, self.splits

    def _pool_for_split_key(self, key: SplitKey):
        if key is None:
            return self.task_pool
        return self.task_pool[key]

    def determine_split_idx(self, key: SplitKey = None, split: Optional[List[float]] = None):
        ''' Helper function for data spliting that returns split indices.
        '''
        pool = self._pool_for_split_key(key)
        split = split if split is not None else self.splits[key] if isinstance(self.splits, dict) else self.splits
        self.total_length = self.determine_len(pool)

        self.split_idx:List[int] = [
                int(self.total_length * split[0]),
                int(self.total_length * (split[0] + split[1])),
                ]
        return self.split_idx

    def determine_len(self, data):
        ''' Helper function that determines length of given data.
        This function must be changed, if the dataset cannot be measured by a simple len operator.
        '''
        return len(data)

    def split_pool(self):
        """ Splits the task pool into training, validation and testing tasks,
        and creates the respective torch datasets."""
        for key, split in self.iter_splits():
            split_idx = self.determine_split_idx(key, split)
            pool = self._pool_for_split_key(key)
            train_pool = pool[:split_idx[0]]
            val_pool = pool[split_idx[0]:split_idx[1]]
            test_pool = pool[split_idx[1]:]

            if key is None:
                self.param_ranges['train'] = train_pool
                self.param_ranges['val'] = val_pool
                self.param_ranges['test'] = test_pool
            else:
                self.param_ranges['train'][key] = train_pool
                self.param_ranges['val'][key] = val_pool
                self.param_ranges['test'][key] = test_pool

    # @final
    # def create_datasets(self):
    #     ''' Creates pytorch datasets for the given problem.
    #     This functions does not have to be changed as long as self.param_ranges is set correctly by split_pool.
    #     '''
    #     if PipelineParams.get_config().baseline:
    #         self.train_dataset = make_baseline_dataset(self.dataset_class)
    #         self.train_dataset = self.train_dataset(pool=self.param_ranges['train'])
    #     else:
    #         self.train_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['train'])
    #     self.val_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['val'])
    #     self.test_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['test'])


    # @final
    # def create_datasets(self):
    #     ''' Creates pytorch datasets for the given problem.
    #     Baseline and meta-learning use the SAME task dataset; they differ only in
    #     the train collate (window pooling) and in the update rule.
    #     '''
    #     self.train_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['train'])
    #     self.val_dataset:   SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['val'])
    #     self.test_dataset:  SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['test'])
    @final
    def create_datasets(self):
        if PipelineParams.get_config().baseline:
            baseline_cls = make_baseline_dataset(self.dataset_class)
            self.train_dataset: SupervisedLearningDataset = baseline_cls(pool=self.param_ranges['train'])
        else:
            self.train_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['train'])

        # val/test stay task-based (M-way episodes), unaffected by baseline flag
        self.val_dataset:  SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['val'])
        self.test_dataset: SupervisedLearningDataset = self.dataset_class(pool=self.param_ranges['test'])
    # @final
    # def train_dataloader(self):
    #     if PipelineParams.get_config().baseline:
    #         # collate_fn   = default_collate
    #         collate_fn = make_window_collate(
    #         windows_per_step=self.windows_per_step,
    #         use_query=self.use_query_windows)
    #         # self.N *= 10
    #     else:
    #         collate_fn = self.collate_fn
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.N, shuffle=True,
    #         num_workers=self.num_workers, pin_memory=self.pin_memory,
    #         collate_fn=collate_fn,
    #         prefetch_factor=2, persistent_workers=True,
    #     )
    @final
    def train_dataloader(self):
        if PipelineParams.get_config().baseline:
            collate_fn = default_collate
            batch_size = min(self.windows_per_step, len(self.train_dataset))
        else:
            collate_fn = self.collate_fn
            batch_size = self.N

        kwargs = dict(
            batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = True

        return DataLoader(self.train_dataset, **kwargs)
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=batch_size, shuffle=True,
        #     num_workers=self.num_workers, pin_memory=self.pin_memory,
        #     collate_fn=collate_fn,
        #     prefetch_factor=2, persistent_workers=True,
        # )
    @final
    def val_dataloader(self):
        N_val = self.config.get("N_val", 10000)
        return DataLoader(
            self.val_dataset,
            batch_size=min(N_val, self.val_dataset.__len__()),
            drop_last=False,
            shuffle=False,
            num_workers=0, pin_memory=False, # No pin_memory needed
            collate_fn=self.collate_fn
        )

    @final
    def test_dataloader(self):
        ''' Manually creates the pool of test tasks out of the given test dataset.
        '''
        N_test = self.config.get("N_test", 10000)
        return DataLoader(
            self.test_dataset,
            batch_size=min(N_test, self.test_dataset.__len__()),
            drop_last=False,
            shuffle=False,
            num_workers=0, pin_memory=False,
            collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        ''' Collate function used for the dataloaders.
        Pytorch uses it to bundle samples into a single batch tensor.
        However, for meta-learning, one wants an iteratable rather than one single tensor.
        '''
        return batch
