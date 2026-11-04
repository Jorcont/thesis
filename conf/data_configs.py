''' Contains data configurations that are embedded in the sampler configs.'''

from dataclasses import dataclass, field
from typing import List, Literal, Union, Dict, Optional
from conf.ConfigBase import ConfigBase
from utils.path_utils import template_to_path
from pathlib import Path

@dataclass
class DataConfig(ConfigBase):
    """ Base class for task configurations.

        Attributes:
            name (optional): Sets the task type.
            config (optional): A dictionary of arguments to be passed to the sampler constructor.

    Whenever a new use case joins this repo and its pipeline, it is expected to have its own dataset class.
    In this case, the user must write his own sub-config for data and sampler.
    This means to:
        1. Create another "...DataConfig" class like below. Those can inherit from one of the already existing ones.
        This step can be skipped if one of the existing sub-configs already suffices.
        2. Add the name to the possible names below.
        3. Add an if-statement in the post_init that creates the DataConfig based on your new name and sub-config.
    Current existing sub-configs are
        1. SupervisedDataset: The main parent class for each Dataset used within this pipeline.
        2. ImageDataset: Specifically created for loading images, e.g. from hugging face.
        3. TimeSeriesDataset: Specifically created to work with pytorch_forecasting's TimeSeriesDataset and all corresponding pytorch_forecasting models.
    """
    name: Literal['SupervisedLearning', 'PVTimeSeries'
            'Grid', 'TimeSeries','Image', 'Audio'] = 'SupervisedLearning'
    config: dict = field(default_factory=dict)
    train_ratio: float = field(default=None)

    def __post_init__(self):
        if self.name == "TimeSeries":
            # Here, it should be modified.
            # TODO: No self, but rather just TimeSeriesDataConfig(**self.config)
            self.config = TimeSeriesDataConfig(**self.config)
        elif (self.name == "Grid") or (self.name == "SupervisedLearning"):
            # TODO: Same here as done above
             self.config = SupervisedLearningDataConfig(**self.config)
        elif (self.name == "Image") or (self.name == "Audio"):
            # TODO: Same here as done above
            self.config = ImageDataConfig(**self.config)
        elif self.name == "PVTimeSeries":
            pass # config dict consumed directly by PVTimeSeriesDatasetV2

@dataclass
class TimeSeriesDataConfig:
    ''' Dataset configuration for timeseries data handled with pytorch_forecasting's TimeSeriesDataset.
    This config is used within this pipeline's TimeSeriesDataset class to create the task sampling and then modified to be used as config
    for pytorch_forecasting's data handling.

    Attributes:
        time_idx: (Mandatory) Specifies the name of the column containing the time indices.
        target (required): column denoting the target or list of columns denoting the target (categorical or continous).
        group_ids: list of column names identifying a time series. This means that the group_ids identify a sample
                together with the time_idx. If you have only one timeseries, set this to the name of column that is constant.
        max_prediction_length: Maximum length of the look-ahead-window. Specifies maximum number of time steps to predict.
        max_encoder_length: Maximum length of the look-back-window. Specifies maximum number of time steps of the history
                to use as input to the model for each instance.
        min_prediction_length: Minimum length of the look-ahead-window. Specifies minimum number of time steps to predict.
        min_encoder_length: Minimum length of the look-back-window. Specifies minimum number of time steps of the history
                to use as input to the model for each instance.
        static_categoricals: list of categorical variables that do not change over time, entries can be also lists
                which are then encoded together (e.g. useful for product categories)
        static_reals: list of continuous variables that do not change over time
        time_varying_known_categoricals: list of categorical variables that change over time and are known in the
                future, entries can be also lists which are then encoded together (e.g. useful for special days or
                promotion categories)
        variable_groups: dictionary mapping a name to a list of columns in the data. The name should be present in a
                categorical or real class argument, to be able to encode or scale the columns by group. This will
                effectively combine categorical variables is particularly useful if a categorical variable can have
                multiple values at the same time. An example are holidays which can be overlapping.
        time_varying_known_reals: list of continuous variables that change over time and are known in the future (e.g.
                price of a product, but not demand of a product)
        time_varying_unknown_reals: list of continuous variables that change over time and are not known in the future.
                You might want to include your target here.
        time_varying_unknown_categoricals: list of categorical variables that change over time and are not known in the
                future. You might want to include your target here.
        variable_groups: dictionary mapping a name to a list of columns in the data.
        lags: dictionary mapping a name to a list of lags for the respective variable.
        add_relative_time_idx: if to add a relative time index as feature (i.e. for each sampled sequence, the index will
                range from -encoder_length to prediction_length)
        add_target_scales: if to add scales for target to static real features (i.e. add the center and scale of the
                unnormalized timeseries as features)
        add_encoder_length: if to add decoder length to list of static real variables.
    '''
    train_ratio: float = 0.5 # TODO: Use this and not above
    time_idx: str = None
    target: Union[List[str], str] = None
    group_ids: List[str] = field(default_factory=list)
    # Prediction and encoder lengths:
    max_prediction_length: int = 1
    max_encoder_length: int = 12
    min_prediction_length: int = 1
    min_encoder_length: int = 10
    # Feature lists grouped by static/time_varying, categorical/real and known/unknown:
    static_categoricals: List[str] = field(default_factory=list)
    static_reals: List[str] = field(default_factory=list)
    time_varying_known_categoricals: List[str] = field(default_factory=list)
    time_varying_known_reals: List[str] = field(default_factory=list)
    time_varying_unknown_reals: List[str] = field(default_factory=list)
    time_varying_unknown_categoricals: List[str] = field(default_factory=list)

    variable_groups: Dict[str, List[str]] = field(default_factory=dict)
    lags: Dict[str, List[int]] = field(default_factory=dict)

    # Indexing:
    add_relative_time_idx: bool = False
    add_target_scales: bool = False
    add_encoder_length: bool = False

    def __post_init__(self) -> None:
        ''' Sets defaults in case of missing or None variables.'''
        if self.target is None:
            raise KeyError("The data config must contain a target key.")
        if self.min_encoder_length is None:
            self.min_encoder_length = self.max_encoder_length
        if self.min_encoder_length != self.max_encoder_length:
            self.add_encoder_length = True
        if len(self.group_ids) == 0:
           self.group_ids = ['group_id']

@dataclass
class SupervisedLearningDataConfig:
     ''' Minimal example of a data config.
     It simply determines the train/test ratio within a task and sets a path
     where to use the data from.
     '''
     data_path: str
     train_ratio: float = 0.5

     def __post_init__(self):
        self.data_path: Path = template_to_path(self.data_path)

@dataclass
class ImageDataConfig(SupervisedLearningDataConfig):
        ''' Class for Prototype learning data.
        Defines data shape by determining number of shots, number of classes etc.
        
        Args:
                m_way: Number of classes to be considered for one single task.
                K_shots: Number of samples per class.
                    Note: This number will be divided into source and query data.
                    input_name: Name of input column in the data found in data_path.
                    label_name: Analog to input_name.
        '''
        m_way: int = 3
        K_shots: int = 10
        input_name: str = "img"
        label_name: str = "fine_label"
        cifar: bool = False
