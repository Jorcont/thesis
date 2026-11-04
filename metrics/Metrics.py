''' Basic evaluation metrics for general meta-learning.'''

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from conf.data_configs import DataConfig

METRIC_REGISTRY = {
    "accuracy": MulticlassAccuracy,
    "precision": MulticlassPrecision,
    "recall": MulticlassRecall,
    "F1": MulticlassF1Score
}

class Metrics:
    ''' This class receives a dictionary of task-specific
    training and testing losses and evaluates the corresponding performance based
    on the metrics configured within this class.'''
    def __init__(self, config):
        self.device = config.device
        self.calc_generalization = config.generalization
        self.calc_adaptation_speed = config.adaptation_speed
        self.calc_sample_efficiency = config.sample_efficiency
        self.print_results = config.print_results
        self.metric_keys = config.metric_keys
        self.init_metrics()

    def init_metrics(self):
        ''' Initializes task-specific metrics and must potentially rewritten for
        specific use cases.
        Note that any metric must be a dict with keys train and test to work with the other metric functions.'''
        self.task_metrics = {key: {'train': [], 'test': []} for key in self.metric_keys}

        self.tmp_metrics = {}
        for key in self.metric_keys:
            if key not in self.task_metrics:
                raise ValueError("Unknown metric: {key}")
            if key == 'losses':
                continue

            train_acc_metric = METRIC_REGISTRY[key](num_classes=DataConfig.get_config().config.m_way)
            test_acc_metric = METRIC_REGISTRY[key](num_classes=DataConfig.get_config().config.m_way)
            train_acc_metric.to(self.device)
            test_acc_metric.to(self.device)
            self.tmp_metrics[key] = {'train': train_acc_metric, 'test': test_acc_metric}

    def compute_metrics(self, preds, targets, loss, mode='train'):
        ''' Updates all metrics with the given preds and targets
        for the given mode (train or test).
        This example only considers losses.
        '''
        for key in self.metric_keys:
            if key == 'losses':
                current_metric = loss
            else:
                self.tmp_metrics[key][mode].reset()
                self.tmp_metrics[key][mode].update(preds, targets)
                current_metric = self.tmp_metrics[key][mode].compute().item()

            self.task_metrics[key][mode].append(current_metric)


