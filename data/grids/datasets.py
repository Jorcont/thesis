from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform, Compose

from data.grids.custom_transforms import NormalizeVoltage

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class NetGraph:
    """Represents a network model in the form of a graph.

    Attributes
    ----------
        node_features : pandas MultiIndex dataframe
            node feature matrix of the graph with shape [num_nodes, num_features x num_samples]
            the indices of the dataframe are node indices and columns are a MultiIndex with each
            feature consisting of a set of samples
        node_target : pandas MultiIndex dataframe
            node target matrix for training with shape [num_nodes, num_target x num_samples]
            the indices of the dataframe are node indices and columns are a MultiIndex with each
            target consisting of a set of samples
        adjacency_matrix : pandas dataframe
            adjacency matrix of the graph with shape [num_edges, 2]
            the indices of the dataframe are edge indices and columns are
            from and to node indices of edges

    """

    node_features: pd.DataFrame
    edge_features: pd.DataFrame | None
    node_target: pd.DataFrame
    adjacency_matrix: pd.DataFrame

    def __post_init__(self) -> None:
        # test node indices
        if set(self.node_features.index) != set(self.node_target.index):
            msg = "Node indices are inconsistent"
            raise ValueError(msg)

    @property
    def node_idx(self) -> NDArray[np.int32]:
        return self.node_features.index.to_numpy()

    @property
    def num_features(self) -> int:
        return self.node_features.columns.levshape[0]

    @property
    def num_targets(self) -> int:
        return self.node_target.columns.levshape[0]

    @property
    def num_samples(self) -> int:
        return self.node_features.columns.levshape[1]

    @property
    def num_node(self) -> int:
        return len(self.node_features.index)



class GraphDataset(Dataset):
    """Creates a Pytorch Geometric based graph dataset.

    Attributes
    ----------
    netgraph : NetGraph
        represents a network model in the form of a graph;
        contains graph data and training samples for
        creating a PyG based graph dataset
    transforms : Union[BaseTransform, List[BaseTransform]]
        represents needed transformations for the dataset
    """

    def __init__(self, netgraph: NetGraph,
                 transforms: BaseTransform | list[BaseTransform] = NormalizeVoltage) -> None:
        super().__init__()
        """Initialize the Graph Attention model.
            Parameters
            ----------
            netgraph : NetGraph
                represents a network model in the form of a graph;
                contains graph data and training samples for
                creating a PyG based graph dataset
            transforms : BaseTransform | list[BaseTransform]
                Transforms to be applied to the data

            Returns
            -------
            None
        """

        self.netgraph = netgraph

        self.feature_names = netgraph.node_features.columns.levels[0].to_numpy()

        self.target_names = netgraph.node_target.columns.levels[0].to_numpy()

        self.num_samples = self.netgraph.num_samples

        self.node_features = (torch.from_numpy(self.netgraph.node_features.values).float().flatten()
                              .view(-1, self.netgraph.num_features, self.num_samples).permute((2, 0, 1)))

        self.node_target = (torch.from_numpy(self.netgraph.node_target.values).float().flatten()
                            .view(-1, self.netgraph.num_targets, self.num_samples).permute((2, 0, 1)))

        # extract the edge ids from the multi index dataframe column header
        if len(netgraph.node_features.columns[0]) == 3:  # noqa: PLR2004
            self.edge_ids = [None if pd.isna(col[2]) else int(col[2])
                             for col in netgraph.node_features.columns.to_numpy()]
        else:
            self.edge_ids = []

        edges = self.netgraph.adjacency_matrix.to_numpy().T
        reverse_edges = np.array([edges[1], edges[0]])
        self.edge_index = torch.from_numpy(
            np.concatenate((edges, reverse_edges), axis=1),
        )

        # compose and apply data transformations
        self.transforms = Compose(transforms) if isinstance(transforms, list) else transforms()
        self.transforms(self)

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def get(self, idx: int) -> Data:
        """Load the single graph from the samples at index idx.

        Parameters
        ----------
        idx : int
            for indexing into the set of graph samples

        Returns
        -------
        graph : object
            one graph sample as PyG data object

        """
        if self.edge_ids:
            edge_id = self.edge_ids[idx]
            # remove deactivated edge from edge_index for N-1 cases
            if edge_id is not None:
                # get id of reverse edge
                reverse_edge_id = edge_id + self.edge_index.shape[1] // 2 - 1
                # remove the edge at edge_id
                new_edge_index = torch.cat((self.edge_index[:, :edge_id],
                                            self.edge_index[:, edge_id + 1:]),
                                           dim=1)
                # remove the edge at reverse_edge_id
                new_edge_index = torch.cat((new_edge_index[:, :reverse_edge_id],
                                            new_edge_index[:, reverse_edge_id + 1:]),
                                           dim=1)
            # don't change edge_index for the N-0 case, when we have both N-0 and N-1 data
            else:
                new_edge_index = self.edge_index
        # don't change edge_index if only N-0 data is used
        else:
            new_edge_index = self.edge_index

        return Data(x=self.node_features[idx],
                    edge_index=new_edge_index,
                    y=self.node_target[idx])
