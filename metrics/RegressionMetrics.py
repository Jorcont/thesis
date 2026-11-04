"""Regression metrics for PV time series forecasting."""

import torch
from conf.data_configs import DataConfig


class RegressionMetrics:
    """
    Simple regression metrics — replaces classification-based Metrics.
    Tracks MAE and MSE for train/test splits.
    """

    def __init__(self, config):
        self.device              = config.device
        self.calc_generalization = config.generalization        
        self.calc_sample_efficiency = getattr(config, 'sample_efficiency', False)
        self.print_results       = config.print_results
        self.metric_keys         = config.metric_keys
        self.init_metrics()

    # def init_metrics(self):
    #     empty_metric = {"train": [], "test": []}
    #     self.task_metrics = {key: empty_metric.copy() for key in self.metric_keys}

    def init_metrics(self):
        self.task_metrics = {
            key: {"train": [], "test": []} for key in self.metric_keys
        }

    def compute_metrics(self, preds, targets, loss, mode="train"):
        for key in self.metric_keys:
            if key == "losses":
                current_metric = loss
            elif key == "mae":
                current_metric = torch.mean(torch.abs(preds - targets)).item()
            elif key == "mse":
                current_metric = torch.mean((preds - targets) ** 2).item()
            elif key == "rmse":
                current_metric = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
            else:
                current_metric = loss
            self.task_metrics[key][mode].append(current_metric)