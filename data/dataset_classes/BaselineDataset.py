import logging
from typing import Type, Tuple

import torch

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset

logger = logging.getLogger("meta-netze")


def make_baseline_dataset(dataset_cls: Type[SupervisedLearningDataset]):
    ''' Factory that turns a task-based (few-shot) dataset class into a
    baseline (single-sample) dataset class.

    Key difference to the meta case:
        - sample_task(idx) returns only the FEW-SHOT subset of a task and is
          therefore NOT suitable here.
        - For baseline training we need EVERY sample of a task, so a dedicated
          hook `full_task(idx)` must provide the COMPLETE (X, y) of a task.

    The wrapper only changes what a dataset index means:
        - meta:      index -> one few-shot task (X_train, y_train, X_test, y_test)
        - baseline:  index -> one single sample (x, y)

    PyTorch's default_collate (now used by the sampler) stacks the returned
    single samples into (B, ...) batch tensors automatically.

    Usage:
        config['dataset_class'] = make_baseline_dataset(MyDataset)
    '''

    class _BaselineDataset(dataset_cls):
        ''' Auto-generated baseline wrapper (single-sample access).'''

        def full_task(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
            ''' Returns the COMPLETE (X, y) of task `idx` (all samples, no
            few-shot subsampling, no train/test split).

            Default tries `full_task` / `load_full_task` on the wrapped class
            if defined; otherwise it MUST be implemented per use case, because
            sample_task only yields the few-shot subset.
            '''

            fn = getattr(super(), "full_task", None)
            if callable(fn):
                return fn(idx)
            raise NotImplementedError(
                f"{dataset_cls.__name__} must provide full task data for "
                "baseline training. Implement `full_task(idx)` (or "
                "`load_full_task(idx)`) returning the COMPLETE (X, y) of a "
                "task. sample_task() only returns the few-shot subset and "
                "cannot be used here."
            )
        def __len__(self):
            ''' Total number of individual samples across the whole pool.'''
            return len(self.index_map)

        def init_data(self):
            ''' Runs the wrapped init_data, then loads the FULL data of every
            task once, caches it, and builds a flat (task_idx, sample_idx) map
            so each dataset index addresses exactly one sample.'''
            super().init_data()
            self._task_cache: dict = {}
            self.index_map: list = []
            n_tasks = super().__len__()      # wrapped __len__ counts tasks
            for t_idx in range(n_tasks):
                X, y = self.full_task(t_idx)
                self._task_cache[t_idx] = (X, y)
                n_samples = (
                    X[0].shape[0] if isinstance(X, (tuple, list)) else X.shape[0]
                )
                for s_idx in range(n_samples):
                    self.index_map.append((t_idx, s_idx))
            logger.info("BaselineDataset(%s): %d tasks -> %d samples "
                        "(full task data)",
                        dataset_cls.__name__, n_tasks, len(self.index_map))                
    
        def __getitem__(self, idx):
            ''' Returns a single (x, y) sample. default_collate stacks these
            into (B, ...) batch tensors.
            NOTE: __getitem__ is @final in the base class, but @final is only
            a static-analysis hint and not enforced at runtime, so the
            override is valid.'''
            t_idx, s_idx = self.index_map[idx]
            # X, y = self.full_task(t_idx)
            if t_idx not in self._task_cache:
                self._task_cache[t_idx] = self.full_task(t_idx)
            X, y = self._task_cache[t_idx]
            if isinstance(X, (tuple, list)):
                return tuple(x[s_idx] for x in X), y[s_idx]
            return X[s_idx], y[s_idx]

    _BaselineDataset.__name__ = f"Baseline{dataset_cls.__name__}"
    _BaselineDataset.__qualname__ = _BaselineDataset.__name__
    return _BaselineDataset
