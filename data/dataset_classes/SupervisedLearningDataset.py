''' Default pytorch Dataset class for Supervised Learning Task Sampling.'''

import logging
from typing import final, Dict, Any, Tuple
import random
import torch
from torch.utils.data import Dataset

from conf.generator_configs import GeneratorConfig
from conf.data_configs import DataConfig

logger = logging.getLogger("meta-netze")

class SupervisedLearningDataset(Dataset):
    ''' This is the base meta-learning dataset class.
    It functions as the connection to pytorch_lightning and hence contains get_item and len methods.
    Here, a batch is a batch of meta-learning tasks, instead of a batch of samples (as in ordinary problems).
    For details on pytorch_lightning logic, review lightning_configs.py and pytorch_lightning trainer documentations.
    Important Formulae:
        lightning epoch = dataset.len samples (samples = meta-learning tasks).
        number of steps/episodes per epoch = dataset.len / batch_size 
        Number of update steps per epoch = dataset.len / batch_size / lightning's accumulate_grad_batches
        batch = batch_size many calls of dataset.get_item
        See LightningMAMLModule.learn_task_pool for details on how batches are used during training.
    In order to create use-case-specific datasets, inherit from this class and try to change
    nothing but sample_task, len and split_data methods, if possible.
    '''
    @final
    def __init__(self, pool, train=False):
        super().__init__()
        self.pool = pool

        # TODO: Maybe change this to
        self.data_config = DataConfig.get_config()#.config
        # self.train_ratio: float = self.data_config.train_ratio
        self.train_ratio: float = self.data_config.train_ratio if self.data_config is not None else 0.7  # ← add fallback


        # Variable "train" has no effect in this version,
        # but is required for other child classes.
        self.train = train
        self.init_data()
        # Set generator if it exists.
        # Jorge's case: no data generated, everything is given. = None
        self.generator = GeneratorConfig.get_config()
        if self.generator is not None:
            logger.info('Using the configured data generator in the dataset.')
            self.generator = self.generator.generator

    def init_data(self):
        ''' Extracts all required information from the data_config
        that is needed for task sampling/creation.
        Also use this to set attributes, which are needed for your use case, instead of changing the class init.
        In this version, this function simply does nothing.
        '''
        pass

    @final
    def set_train_flagg(self, train_flagg):
        ''' Changes self.train if required.'''
        self.train = train_flagg

    def sample_task(self, idx):
        ''' Placeholder for task sampling.
        Important: The sampled data must be split afterwards.
        The given index determines which task to sample. Since pytorch_lightning is used
        for data handling, idx is random during meta-training and in order for val and test.
        Always place your own logic in this class, when facing a new use case.
        '''
        return self.pool[idx]

    def split_data(self, data: Any) -> Tuple[Any, Any, Any, Any]:
        """ Splits the given data into two parts, the train and test data
        Besides sample_task, this is the function that is most likely to change for a new use case.
        using self.train_ratio.
        Returns:
            X_train, y_train: Training data pairs.
            X_test, y_test: Test data pairs.
        """
        return None, None, None, None

    def __len__(self):
        ''' Determines the size of the dataset.
        In meta-learning, it (theoretically) defines the number
        of different tasks per task family. Practically, it is responsible to determine how
        long one pytorch epoch is. However, it often has to be set manually by the user.'''
        return len(self.pool)

    @final
    def __getitem__(self, idx):
        ''' This function is nt expected to be changed, as both functions called within, are changables.
        It defines, that one sample (from pytorch_lightning perspective) is one task (from meta-learning perspective).
        Pytorch_lightning will use this method "batch_size" many times (see supervisedsampler dataloaders for batch_size) in order to create a batch.
        See the class description for more details.

        Important: data must be tupel of X and y, while split_data returns X_train, y_train, X_test, y_test.
        The latter represents a full task's data.'''
        data = self.sample_task(idx)
        return self.split_data(data)
    
    def full_task(self, idx):
        ''' This function must return the full task's data.
        It is ment to be used for baseline training, where no few-shot problems are assumed.
        Changing this function properly results in automatic adjustment of dataset to baseline case.
        Returns: X,y: batch of full task data.
        '''
        X, y = self.pool(idx)
        return X, y

    ''' The following functions are ment to be used in extended_inner_learning function of LightningMAMLModule.
    They serve as the connection between use-case specific implementation and the unchangable pipeline logic.
    '''

    def copy_batch(self, X, y):
        ''' Creates a copy of the given batch.
        Highly depends on the type of use case-specific data.'''
        X_copy = X.clone() if hasattr(X, "clone") else X.copy()
        y_copy = y.clone()if hasattr(y, "clone") else y.copy()
        return X_copy, y_copy

    def add_one_sample(self, batch, i, full_batch):
        ''' Adds sample with index i of the given full_batch
        to the given batch. Highly depends on the data type used.'''
        return torch.cat([batch, full_batch[i:i+1]], dim=0)

    def get_emptyX(self, X):
        ''' Returns an empty object of the type of input data X.
        X is supposed to be the original dataset.'''
        return torch.empty(0,*X.shape[1:], device= X.device)

    def get_emptyY(self, y):
        ''' Analog to get_emptyX.'''
        return torch.empty(0,*y.shape[1:], dtype=torch.long, device= y.device)

    def full_task(self, idx):
        ''' This function must return the full task's data.
        It is ment to be used for baseline training, where no few-shot problems are assumed.
        Changing this function properly results in automatic adjustment of dataset to baseline case.
        Returns: X,y: batch of full task data.
        '''
        X, y = self.pool(idx)
        return X, y
