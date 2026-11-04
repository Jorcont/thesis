import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform


class NormalizeVoltage(BaseTransform):
    """Transform to normalize the voltage degree of the dataset."""

    def forward(self, data: Dataset) -> Dataset:
        """Transform the voltage degree of the dataset to be normalized.

        Parameters
        ----------
        data : Dataset
            Dataset to be transformed

        Returns
        -------
        data : Dataset
            Transformed dataset with its voltage degree normalized
        """
        if "va_degree" in data.target_names:
            idx = np.where(data.target_names == "va_degree")[0][0]
            data.node_target[:, :, idx] += 360
            data.node_target[:, :, idx] /= 360

        if "va_degree_dc" in data.target_names:
            idx = np.where(data.target_names == "va_degree_dc")[0][0]
            data.node_target[:, :, idx] += 360
            data.node_target[:, :, idx] /= 360

        return data
