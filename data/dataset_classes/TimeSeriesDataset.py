''' Default pytorch Dataset class for Supervised Learning Task Sampling with timeseries data.'''

from typing import Any, Dict, Tuple
from typing_extensions import override
import random
from pytorch_forecasting.data import timeseries as ts
import torch

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from conf.data_configs import DataConfig

class TimeSeriesDataset(SupervisedLearningDataset):
    @override
    def init_data(self):
        ''' Just deleting keys from data_config because it needs
        to have a certain shape for pytorch_forecasting's TimeSeriesDataset.'''
        self.data_config = DataConfig.to_dict()
        print(f"Data config: {self.data_config}")
        print(f"These keys: {self.data_config.keys()}")
        del self.data_config['train_ratio']

    def sample_task_id(self):
        ''' Samples a task id from given pool.'''
        task_id = {}
        for key in self.pool.keys():
            task_id[key] = random.uniform(self.pool[key][0], self.pool[key][1])

        return task_id

    @override
    def sample_task(self):
        ''' For this special use case, sampling a task means to
        generate data with the physical model for a certain task_id=parameter set.
        '''
        task_id = self.pool[idx]
        task_id = self.sample_task_id()
        self.generator.set_params(task_id)
        self.generator.reset()
        task_data = self.generator.generate_data()

        self.data = ts.TimeSeriesDataSet(
            data = task_data,
            predict_mode = False,
            **self.data_config
            )
        total = task_data.shape[0]
        self.data_loader = self.data.to_dataloader(
            train=True, # Every batch will always be half train, half val.
            batch_size=total,        # full dataset in one batch
            shuffle=False,
            num_workers=0
        )

        (x_dict, (y_tensor, _)) = next(iter(self.data_loader))
        return x_dict, y_tensor

    @override
    def split_data(self,
                   data: Tuple[Dict[str, torch.Tensor], torch.Tensor],
                   train_ratio: float = None) -> Tuple[Any, Any, Any, Any]:
        """ Splits the task data into two parts, the train and test data.
        Returns:
            X_train, y_train: Training data pairs.
            X_test, y_test: Test data pairs.
        """
        X, y = data

        batch_size = y.shape[0]
        split = int(batch_size * self.train_ratio)

        self.X_train: Dict[str, torch.Tensor] = {}
        self.X_test: Dict[str, torch.Tensor] = {}
        for key, value in X.items():
            # TODO: Consider here using slicing, since we have the split. VERY SAFE.
            self.X_train[key], self.X_test[key] = torch.split(value, split, dim=0)

        self.y_train, self.y_test = torch.split(y, split, dim=0)

        return self.X_train, self.y_train, self.X_test, self.y_test

    @override
    def __len__(self):
        return 1000

    # Todo: Not sure if this is needed anymore.
    def reshape_and_move_tensor(self, t: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Moves tensor to device. No reshaping in this version."""
        if isinstance(t, Dict):
            return {k: self.reshape_and_move_tensor(v) for k, v in t.items()}
        elif isinstance(t, torch.Tensor):
            return t.to(self.device)
        else:
            raise TypeError(f"Parameter t set to unsupported type {type(t)}. Use torch.Tensor or Dict[str, torch.Tensor] instead.")
