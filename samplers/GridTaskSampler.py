''' Task-Sampler child class for grid use case.'''

from typing_extensions import override
import os

from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler

class GridTaskSampler(SupervisedLearningTaskSampler):
    @override
    def create_task_pool(self):
        ''' Creates the task pool from which training, validation and test grid tasks are sampled.
        It searches for folders in the given path. Each folder stands for a task and is assumed to contain
        the respective files (see GridTask.import_data for details).'''
        self.data_path:str = self.data_config['data_path']
        self.task_pool = [os.path.abspath(os.path.join(self.data_path, f))
            for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]

    def set_number_of_tasks(self):
        ''' Determine N_val and N_test as they are not given.
        '''
        self.N_val = len(self.val_dataset.pool)
        self.N_test = len(self.test_dataset.pool)



''' Task in Grid context.'''
# Todo: To be specified what task exactly to perform.

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import override

from data.grids.datasets import NetGraph, GraphDataset
from tasks.SupervisedLearningTask import SupervisedLearningTask

logger = logging.getLogger(__name__)

class GridTask(SupervisedLearningTask):
    @override
    def load_and_process_data(self):
        ''' This is the adjusted data loading and preparation for grid tasks.
        '''
        self.data = self.import_data(
            root=self.task_id,
            scoop=self.config['scoop'],
            type_pf=self.config['type_pf'],
            add_nminus1_data=self.config['add_nminus1_data']
        )

        self.train_data, self.test_data = self.split_data(data =self.data, split=self.config['data_split'])

        self.training_dataset = GraphDataset(netgraph=self.train_data)
        self.X_train = self.reshape_and_move_tensor(self.training_dataset.node_features)
        self.y_train = self.reshape_and_move_tensor(self.training_dataset.node_target)

        self.test_dataset = GraphDataset(netgraph=self.test_data)
        self.X_test = self.reshape_and_move_tensor(self.test_dataset.node_features)
        self.y_test = self.reshape_and_move_tensor(self.test_dataset.node_target)

    @override
    def import_data(
            self,
            root: str | Path,
            scoop: int | None = None,
            type_pf: str = "ac",
            add_nminus1_data: bool = False,
    ) -> NetGraph:
        """Import graph data from pickle files.

        Parameters
        ----------
        root : Path
            dir where node features, edge features, node targets, adjacency matrix are stored
            as pickle files
        split : str
            either test or train. Defines whether to load training or test data. Defaults to train.
        scoop : Int
            number of samples to select from the datasets (useful for training locally when datasets are large)
        type_pf : str
            which data type to load: "ac", "dc" or "all" power-flow values
        add_nminus1_data : bool
            whether n-1 data should be used or not.

        Returns
        -------
        training_data: NetGraph
            NetGraph object for training data
        validation_data: NetGraph
            NetGraph object for validation data

        """

        root = Path(root)

        # Check if n-1 data is available
        n_files = len(os.listdir(root))
        if add_nminus1_data and n_files <= 3:
            logger.warning("Warning: 'add_nminus1_data' is %s but there is no n-1 data available.", add_nminus1_data)

        if not add_nminus1_data:  # only n-0 data
            nf = pd.read_pickle(root / "node_features.p").fillna(0.00)  # noqa: S301
            nt = pd.read_pickle(root / "node_target.p").fillna(0.00)  # noqa: S301

        else:  # all n-0 and n-1 data
            nf = self.load_nminus1_data(root, file="node_features")
            nt = self.load_nminus1_data(root, file="node_target")

        am = pd.read_pickle(root / "adjacency_matrix.p")  # noqa: S301

        if type_pf == "ac":
            nf_cols = [col for col in nf.columns if not str(col[0]).endswith("_dc")]
            nt_cols = [c for c in nt.columns if not str(c[0]).endswith("_dc")]

            nf = nf.loc[:, nf_cols]
            nf.columns = nf.columns.remove_unused_levels()

            nt = nt.loc[:, nt_cols]
            nt.columns = nt.columns.remove_unused_levels()

        elif type_pf == "dc":
            nf_cols = [col for col in nf.columns if str(col[0]).endswith("_dc")]
            nt_cols = [c for c in nt.columns if str(c[0]).endswith("_dc")]

            nf = nf.loc[:, nf_cols]
            nf.columns = nf.columns.remove_unused_levels()

            nt = nt.loc[:, nt_cols]
            nt.columns = nt.columns.remove_unused_levels()

        elif type_pf == "all":
            pass

        else:
            message = 'Invalid type parameter provided. Please provide one of the following types: "ac", "dc", "all"'
            raise ValueError(message)

        if scoop is not None:
            nf, nt = self._scoop(nf=nf, nt=nt, scoop=scoop)

        return NetGraph(node_features=nf,
                        node_target=nt,
                        edge_features=None,
                        adjacency_matrix=am)

    @override
    def split_data(
            self,
            data: NetGraph,
            split: float = 0.2,
    ) -> tuple[NetGraph, NetGraph]:
        """Split graph data into two parts according to given split fraction.

        Parameters
        ----------
        data: NetGraph
            Netgraph containing samples to be split
        split: float
            fraction of samples from the dataset to be assigned to the validation set

        Returns
        -------
        training_data: NetGraph
            NetGraph object for training data
        validation_data: NetGraph
            NetGraph object for validation data

        """
        if len(data.node_features.columns.levels) == 2:  # n-0 data
            n_train = np.floor(len(data.node_features.columns.levels[-1]) * (1 - split)).astype(int)
            n_val = np.floor(len(data.node_features.columns.levels[-1]) * split).astype(int)
        elif len(data.node_features.columns.levels) == 3:  # n-1 data
            n_train = np.floor(len(data.node_features.columns.levels[-2]) * (1 - split)).astype(int)
            n_val = np.floor(len(data.node_features.columns.levels[-2]) * split).astype(int)
        else:
            msg = (f"Unexpected data format: Expected 2 or 3 column levels in node_features, "
                   f"but got {len(data.node_features.columns.levels)}.")
            raise ValueError(msg)
        nf_train, nt_train = self._scoop(nf=data.node_features, nt=data.node_target, scoop=n_train)
        nf_val, nt_val = self._scoop(nf=data.node_features, nt=data.node_target, scoop=n_val, start_idx=n_train)

        training_data = NetGraph(node_features=nf_train,
                                 node_target=nt_train,
                                 edge_features=None,
                                 adjacency_matrix=data.adjacency_matrix)

        val_data = NetGraph(node_features=nf_val,
                            node_target=nt_val,
                            edge_features=None,
                            adjacency_matrix=data.adjacency_matrix)

        return training_data, val_data

    def _scoop(
            self,
            nf: pd.DataFrame,
            nt: pd.DataFrame,
            scoop: int,
            start_idx: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scoop samples from the dataset."""
        nf_idf = nf.columns.levels[0]
        nt_idf = nt.columns.levels[0]

        nlevels = nf.columns.nlevels
        sample_labels = nf.columns.levels[nlevels - 1]
        sidx_slice = sample_labels[start_idx: start_idx + scoop]

        if nlevels == 2:
            nf_sliced = nf.loc[:, (nf_idf, sidx_slice)]
            nt_sliced = nt.loc[:, (nt_idf, sidx_slice)]
        elif nlevels == 3:
            nf_sliced = nf.loc[:, (nf_idf, slice(None), sidx_slice)]
            nt_sliced = nt.loc[:, (nt_idf, slice(None), sidx_slice)]
        else:
            raise ValueError(f"Expected 2 or 3 column levels, got {nlevels}.")

        nf_sliced.columns = nf_sliced.columns.remove_unused_levels()
        nt_sliced.columns = nt_sliced.columns.remove_unused_levels()

        return nf_sliced, nt_sliced

    def load_nminus1_data(root: str | Path,
                          file: str = "node_features",
                          ) -> pd.DataFrame:
        """Import n-0 and n-1 data from pickle files.

        Parameters
        ----------
        root : Path
            dir where node features, edge features, node targets, adjacency matrix are stored
            as pickle files
        split : str
            either test or train. Defines whether to load training or test data. Defaults to train.
        file : str
            either node_features or node_target. Defines whether to load node features or node targets.

        Returns
        -------
        df_concat: pd.DataFrame
            Concatenated multi-index DataFrame that contains all the n-0 and n-1 data.

        """
        root = Path(root)

        # Get all relevant .p files, e.g. "node_features.p", "node_features_4.p", etc.
        files = sorted(f for f in os.listdir(root) if f.startswith(file) and f.endswith(".p"))

        df_list = []
        edge_ids_list = []

        for f in files:
            match = re.search(r"_(\d+)\.p$", f)
            edge_id = int(match.group(1)) if match else -1
            edge_ids_list.append(edge_id)

            df_temp = pd.read_pickle(root / f)  # noqa: S301

            # Re-label columns (var, edge_id, sample_idx)
            unique_vars = df_temp.columns.get_level_values(0).unique()
            var_counters = {var: 0 for var in unique_vars}
            new_cols = []
            for col in df_temp.columns:
                var = col[0]
                sample_idx = var_counters[var]
                var_counters[var] += 1
                new_cols.append((var, edge_id, sample_idx))  # edge_id is now in level 1, sample_idx in position 2

            df_temp.columns = pd.MultiIndex.from_tuples(new_cols)
            df_list.append(df_temp)

        # Concatenate all dataframes side by side
        return pd.concat(df_list, axis=1)
