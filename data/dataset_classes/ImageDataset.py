''' Default pytorch Dataset class for Supervised Task Sampling with Image data.'''

from typing_extensions import override
from typing import Any, Dict, Tuple
import random
from math import comb
from datasets import load_dataset, concatenate_datasets
from typing import List
from collections import defaultdict
from algorithms.PipelineState import PipelineState as state

import torch
from torchvision.transforms.functional import to_tensor

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset
from conf.configs import PipelineParams

class ImageDataset(SupervisedLearningDataset):
    @override
    def init_data(self):
        ''' Extracts all required information from the data_config.
        Note, here data_path means the huggingface ID for the respective dataset.
        If the number of classes per task, M_way, is None, the number of classes per task is arbitrary.
            This yields random sampling of M_way and setting pipeline_state lagg to true accordingly.
        '''
        self.m_way = self.data_config.m_way
        self.K_shots = self.data_config.K_shots
        self.input_name = self.data_config.input_name
        self.label_name = self.data_config.label_name
        self.unique_labels = list(set(self.pool[self.label_name]))
        self.label_to_indices = defaultdict(list)
        labels = self.pool[self.label_name]
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    @override
    def sample_task(self, idx):
        ''' Defines sampling tasks dependent on the information gathered from data config.
        Here, every task is M_way classification for K_shots each class.
        Hence, the total number of samples must be M_way * K_shots.
        '''
        selected_classes = self.sample_class_ids()
        label_feature = self.pool.features[self.label_name]
        #selected_classes_string = [label_feature.int2str(x) for x in selected_classes]
        #print("classes", selected_classes_string)
        
        all_indices = []
        for c in selected_classes:
            sampled = random.sample(self.label_to_indices[c], self.K_shots)
            all_indices.extend(sampled)
        random.shuffle(all_indices)
        dataset = self.pool.select(all_indices)

        X = dataset[self.input_name]
        X = self.preprocess(X)
        y = dataset[self.label_name]
        y = torch.tensor(y, dtype=torch.long)
        return X,y

    def preprocess(self, X):
        ''' Preprocesses X to be a normalized, properly scaled input tensor.
        In this version, it is a simple torch.stack.
        '''
        return torch.stack([to_tensor(img) for img in X])

    def sample_class_ids(self) -> List[int]:
        ''' Samples class ids from the given classes.'''
        if self.m_way is not None:
            return random.sample(self.unique_labels, self.m_way)
        else:
            return random.sample(self.unique_labels, self.sample_num_classes())

    def sample_num_classes(self):
        ''' Randomly samples the number M of classes for one particular task.
        Only triggered, if self.M_way is None.
        '''
        return random.randint(3, 10)

    @override
    def split_data(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Splits the task data into two parts, the train and test data.
        Returns:
            X_train, y_train: Training data pairs.
            X_test, y_test: Test data pairs.
        """
        X, y = data

        unique_classes = y.unique()
        self.X_train, self.y_train = [], []
        self.X_test, self.y_test = [], []

        for i, c in enumerate(unique_classes):
            idx = (y == c).nonzero(as_tuple=True)[0]          # indices for this class
            class_X = X[idx]
            class_y = y[idx]
            batch_size = class_y.shape[0]
            split = int(batch_size * self.train_ratio)

            class_y.fill_(i)
            X_train, X_test= torch.split(class_X, split, dim=0)
            y_train, y_test = torch.split(class_y, split, dim=0)

            self.X_train.append(X_train)
            self.y_train.append(y_train)
            self.X_test.append(X_test)
            self.y_test.append(y_test)

        self.X_train= torch.cat(self.X_train, dim=0)
        self.y_train= torch.cat(self.y_train, dim=0)
        self.X_test= torch.cat(self.X_test, dim=0)
        self.y_test= torch.cat(self.y_test, dim=0)

        # Shuffle the datasets.
        perm = torch.randperm(self.X_train.size(0))
        self.X_train = self.X_train[perm]
        self.y_train= self.y_train[perm]

        perm = torch.randperm(self.X_test.size(0))
        self.X_test= self.X_test[perm]
        self.y_test= self.y_test[perm]

        return self.X_train, self.y_train, self.X_test, self.y_test

    @override
    def __len__(self):
        ''' Returns the number of different classes in this dataset's pool.'''
        return len(self.unique_labels)

    @override
    def full_task(self, idx):
        ''' Returns the full data for one single class.
        Note, that this means the notion "task" to not refer to m-way K-shot.
        Rather, this is for yilding all samples belonging to the class idx.
        '''
        label = self.unique_labels[idx]
        all_indices = self.label_to_indices[label]
        #all_indices = self.unique_labels[idx]
        dataset = self.pool.select(all_indices)
        X = dataset[self.input_name]
        X = self.preprocess(X)
        y = dataset[self.label_name]
        y = torch.tensor(y, dtype=torch.long)

        return X,y
