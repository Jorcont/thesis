''' Default configurations + respective functionalities.'''

import sys
from dataclasses import dataclass, asdict, field
from importlib import import_module
from typing import Literal, Optional, Dict
import torch
from torch import device
from pathlib import Path
import os
import shutil

from conf.ConfigBase import ConfigBase
from conf.config_compare import check_other_config
from conf.loss_configs import LossConfig
from conf.data_configs import DataConfig
from conf.env_configs import EnvConfig
from conf.generator_configs import GeneratorConfig
from conf.model_configs import ModelConfig
from conf.sampler_configs import SamplerConfig
from conf.metrics_configs import MetricsConfig
from evaluation.BasicEvaluations import BasicEvaluations
from metrics.Metrics import Metrics
from algorithms.BaselineWraper import make_ordinary_module
from algorithms.PipelineState import PipelineState as state

from utils.path_utils import template_to_path
import logging

from utils.logging_uitls import setup_logger
logger = logging.getLogger("meta-netze")

@dataclass
class PipelineParams(ConfigBase):
    ''' This is the overal, main config handling all other config creations and any other
    background checks/assignments/creations.
    !!!It should NOT be changed if not particularly neccessary!!!
    Besides setting and distributing attributes, it
    - creates paths for output, checkpoints, results and clones the used yaml-file to those directories
    - creates the module determined by algorithm and lightning atributes. If lightning=True, Lightning modules are utilized.
    - creates the underlying trainer determined by trainer_config (see lightning_configs.py for how to set this sub-config).
    - creates the pipeline object based on lightning flagg. Currently, only lightning pipelines are updated. Ohter pipelines are remainings and might be deleted in the future.
    !!! Note: All sub-configs are expected to be created from the sub_configs.py files.!!!
        These files contain the possibility to created own sub_configs, so that they can be adjusted to the corresponding use case/task distribution.

    Attributes:
    config_path: where to find the used yaml, so that it can be copied into output folder. Is automatically set by main file.
    name (mandatory): The project name. It is used to name all paths for outputs, checkpoints etc.
    logging_level: The logging level used within the whole pipeline.
    mode: The meta-learing class. Currently, only gradient-based meta-learning is implemented.
    algorithm: Which meta-learning algorithm to be used, e.g. MAML, ANIL, RL^2
    lightning: Whether pytorch_lightning should be used. Currently, this is always expected to be true.
    device: The device used for pytorch.
    meta_training_iter: Number of meta-training epochs. If 0, no meta-training is performed.
        Note: One pytorch_lightning epoch is Dataset.__len__ / batch_size, where batch_size episodes of is N_train, N_val or N_test respectively.
        See SupervisedDataset in data folder for more details.
    meta_lr: The meta learning rate. Can be a dict for more precise learning. See LightningMAMLModule for more information.
    fast_lr: The inner/fast learning rate. Same as meta_lr.
    weight_decay: The weight decay of the model.
    adaption_steps: The number of gradient-descent steps in inner learning. Only relevant for gradient-based learning.
    checkpoint_to_load: Specify if any checkpoint shell be loaded for testing or pre-trained training. If None, it is ignored.
    checkpoint_dir: The directory where to SAVE checkpoints. Is not connected with checkpoint_to_load. If not specified, it is automatically created in output folder.
    load_dir: Directory, where to load a chckpoint from, if specified.
    keep_checkpoints: Number of checkpoints to keep during meta-training.
    checkpoint_freq: The frequency of checkpoint creation during meta-training.
    validate_every: Frequency of meta-validation (every X epochs for Lightning).
    print_meta_val: Whether to print/save/evaluate meta-validation results.
    baseline: Whether the module is in baseline mode, where ordinary training takes place instead of meta-training.
    always_init_last_layer (only for classification): If the number of classes per task varries.
            This is determined by the underlying dataset class and yields fresh head init for each single task.
        Automatically sets global state.train flagg accordingly, so that metrics are computed only if needed.
    task_sampler_config: The config for task sampler. See sampler_configs.py and SupervisedLearningSampler in samplers folder for more details.
    model_config: The config for the model. See model_configs.py for more details.
    loss_config: See loss_configs.py for details.
    trainer_config: Only relevant for lightning=True. See Lightning_configs.py for details.
    metrics_config: Sets the metrics to be used. Highly use-case specific and hence likely to change for each task distribution.
    data_config: See data_configs.py and SupervisedDataset in data folder for details.
    env_config (optional): Only relevant for meta-RL. Determines, which environment object(s) are used for RL. See env_configs for details.
    generator_config: Determines if and what data generator is used. This generator is then embedded in the dataset class and functions as the corresponding data sourcg.
        Highly use-case-specific and hence likely to change accordingly.
    evaluations: The meta-evaluation to be calculated after meta-testing.
    # Todo: So far, a very static and redundant structure with no effect on extended_inner_learning.
    '''
    config_path: str
    name: str = "DefaultPipeline"
    logging_level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    mode: Literal["Gradient-based", "Memory-based"] = 'Gradient-based'
    algorithm: Literal["MAML", "ANIL", "RL2"] = 'MAML'
    lightning: bool = False
    baseline: bool = False
    classification: bool = False
    device: Literal["cpu", "cuda", "cuda:0"] | device = 'cpu'
    meta_training_iter: int = 1000
    meta_lr: float = 0.0001
    fast_lr: float | Dict[str, float] = 0.1
    weight_decay: float = 1e-4
    adaptation_steps: int = 5
    first_order: bool = True
    inner_batch_size: int = None
    zero_target_threshold: float = 0.0
    grad_log_every_n_steps: int = 50
    grad_hist_every_n_steps: int = 1000
    log_grad_histograms: bool = False

    # For checkpointing:
    always_init_last_layer: bool = False

    checkpoint_to_load: str = None
    checkpoint_dir: str = None
    load_dir: str = None
    keep_checkpoints: int = 5
    checkpoint_freq: int = 10
    validate_every: int = 100
    print_meta_val: bool = False
    seed: int = None

    task_sampler_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    loss_config: dict = field(default_factory=dict)
    trainer_config: dict = field(default_factory=dict)
    metrics_config: dict = field(default_factory=dict)
    data_config: dict = field(default_factory=dict)
    env_config: Optional[dict] = None
    generator_config: Optional[dict] = None
    evaluations: dict[str, bool] = field(default_factory=lambda: DEFAULT_EVALUATIONS)

    def __post_init__(self) -> None:
        """
        Initialize and finalize the pipeline configuration after dataclass construction.

        This method performs all setup steps that depend on the initialized configuration
        values. It prepares logging, resolves paths, instantiates nested configuration
        objects, loads dataset/model/sampler/loss/metric components, configures the target
        device, and creates the training or evaluation pipeline.

        The method also handles optional reinforcement-learning environments, optional
        data generators, and PyTorch Lightning-specific setup such as trainer creation,
        module creation, checkpoint loading, and data-parallel detection.

        Side Effects:
            - Configures the global logger according to ``logging_level``.
            - Resolves and creates configured output/checkpoint/result paths.
            - Converts ``device`` from a string to ``torch.device``.
            - Replaces raw config dictionaries with instantiated config-derived objects
            where applicable.
            - Dynamically imports and configures the dataset class specified by
            ``data_config["name"]``.
            - Adds the resolved dataset class to ``task_sampler_config["config"]``.
            - Creates and assigns:
                - ``evaluations``
                - ``metrics_config``
                - ``metrics``
                - ``sampler``
                - ``model``
                - ``trainer`` / ``module_class`` / ``module`` when Lightning is enabled
                - ``pipeline``
            - Loads a model from ``checkpoint_to_load`` when Lightning is enabled and a
            checkpoint path is provided.

        Raises:
            KeyError:
                If required keys such as ``data_config["name"]`` or expected nested
                configuration keys are missing.
            ModuleNotFoundError:
                If the dataset module derived from ``data_config["name"]`` cannot be
                imported.
            AttributeError:
                If the expected dataset class is not found in the imported dataset module.
            TypeError:
                If one of the configuration dictionaries contains invalid arguments for
                its corresponding config class.
            RuntimeError:
                If device creation, checkpoint loading, trainer creation, or pipeline
                creation fails.

        Notes:
            The dataset class is resolved dynamically using the naming convention
            ``<data_config["name"]>Dataset`` from the module
            ``data.dataset_classes.<data_config["name"]>Dataset``.
        """
        setup_logger(self.logging_level)
        logging.getLogger().setLevel(self.logging_level)
        logging.getLogger("meta-netze").setLevel(self.logging_level)

        self.resolve_paths()

        self.evaluations: BasicEvaluations = BasicEvaluations(self.evaluations)

        if "config" not in self.task_sampler_config:
            self.task_sampler_config['config'] = {}
        if "config" not in self.model_config:
            self.model_config['config'] = {}

        # Set devices
        self.device = torch.device(self.device)

        # Set RL environment or data generator and include them in task_sampler_config respectively.
        if self.env_config is not None:
            self.env_config = EnvConfig(**self.env_config).__dict__['config']
        if self.generator_config is not None:
            self.generator = GeneratorConfig(**self.generator_config).generator

        # Set dataset class
        # In my case "PVTimeSeriesDataset" --> ok. Import it correctly
        dataset_class_name = self.data_config["name"] + "Dataset"
        module_name = f"data.dataset_classes.{dataset_class_name}"
        dataset_module = import_module(module_name)
        dataset_class = getattr(dataset_module, dataset_class_name)

        # Set data and sampler.
        data_config = DataConfig(**self.data_config)
        if (self.classification) and (data_config.M_way is None):
            # Means no fixed number of classes, i.e. last layer needs a re-init for each task.
            self.always_init_last_layer = True
        # Todo: This looks weird. Why not set this in a global context?
        dataset_class.config = data_config
        self.task_sampler_config["config"]['dataset_class'] = dataset_class

        self.metrics_config = MetricsConfig(**self.metrics_config)
        self.metrics_config.device = self.device
        # self.metrics = Metrics(self.metrics_config)

        # TODO: It will be removed or commented LATER
        if self.metrics_config.name == "Regression":
            from metrics.RegressionMetrics import RegressionMetrics
            self.metrics = RegressionMetrics(self.metrics_config)
        else:
            from metrics.Metrics import Metrics
            self.metrics = Metrics(self.metrics_config)

        if self.lightning:
            devices = self.trainer_config.get("devices", "auto")
            self.task_sampler_config['config']['pin_memory'] = self.device.type == "cuda"
            self.data_parallel = isinstance(devices, int) and devices > 1
            logging.debug(f"data_parallel: {self.data_parallel}")
        self.sampler = SamplerConfig(**self.task_sampler_config).sampler

        self.model = ModelConfig(device=self.device, **self.model_config).model
        LossConfig(**self.loss_config)

        if self.lightning:
            # accelerator = "auto" if str(self.device) in ("cuda", "cuda:0", "gpu") else "cpu"
            self.create_trainer(self.device)
            self.module_class = self.create_module()
            logging.debug(f"My module class is {self.module_class}")
            self.module = self.module_class()
            if self.checkpoint_to_load is not None:
                self.model = self.module_class.load_from_checkpoint(
                        self.checkpoint_to_load,
                        config = self,
                        strict=False,
                        #map_location="cuda:0",   # oder "cpu"
                        #strict=True              # oder False, wenn du Keys geändert hast
                        )

        self.pipeline = self.create_pipeline()

    def resolve_paths(self):
        '''Creates folders for outputs.
        Also sets paths that are not given.
        '''
        if self.name is None:
            raise ValueError("You must define a name for the pipeline.")
        self.output_folder = template_to_path("<output>/" + self.name)
        os.makedirs(self.output_folder, exist_ok=True)

        shutil.copy(self.config_path, self.output_folder / "config_copy.yaml")

        if self.checkpoint_dir is None:
            self.checkpoint_dir: Path = self.output_folder / "checkpoints"
        else:
            self.checkpoint_dir: Path = template_to_path(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.load_dir: Path = template_to_path(self.load_dir) if self.load_dir is not None else self.checkpoint_dir
        check_other_config(self.output_folder / "config_copy.yaml", self.load_dir.parent / "config_copy.yaml")

        if self.checkpoint_to_load is not None:
            self.checkpoint_to_load = self.load_dir / self.checkpoint_to_load # changed +"/"+ to /
            #self.checkpoint_to_load = self.load_dir.__str__() + "/" + self.checkpoint_to_load
        logging.debug(f"checkpoint_dir: {self.checkpoint_dir}")
        logging.debug(f"checkpoint_to_load: {self.checkpoint_to_load}")

    @staticmethod
    def lightning_accelerator(torch_device: torch.device) -> str:
        """Translate a torch device into Lightning's accelerator vocabulary."""
        if torch_device.type == "cuda":
            return "gpu"
        return torch_device.type

    def create_trainer(self, torch_device: torch.device):
        ''' Create trainer with all given information.
        Thereby creates consistency with all global pipeline information, e.g. max_epochs variable.
        See lightning_configs.py for details about possible configurations.
        '''
        from conf.lightning_configs import TrainerConfig
        trainer_conf: TrainerConfig = TrainerConfig(**self.trainer_config)
        trainer_conf.create_checkpoint_callback(self.checkpoint_dir, self.keep_checkpoints, self.checkpoint_freq)
        self.trainer_config = trainer_conf.__dict__
        self.trainer_config["accelerator"] = self.lightning_accelerator(torch_device)
        self.trainer_config["max_epochs"] = self.meta_training_iter
        self.trainer_config["check_val_every_n_epoch"] = self.validate_every

    def create_module(self):
        ''' Creates and returns a pytorch_lightning module object.
        In particular, this function determines which module to use from the algorithms folder.
        '''
        if self.algorithm == 'MAML':
            from algorithms.LightningMAMLModule import LightningMAMLModule
            module_class =  LightningMAMLModule
        elif self.algorithm == 'ANIL':
            from algorithms.LightningANILModule import LightningANILModule
            module_class = LightningANILModule
        elif self.algorithm == 'RL2':
            from algorithms.LightningRL2Module import LightningRL2Module
            module_class = LightningRL2Module
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if self.baseline:
            module_class = make_ordinary_module(module_class)

        return module_class

    def create_pipeline(self):
        ''' Creates the pipeline object based on the defined mode and algorithm.'''
        if self.lightning:
            from algorithms.LightningPipeline import LightningPipeline
            return LightningPipeline(self)
        else: # No lightning usage
            if self.algorithm == "MAML":
                from algorithms.GradientBasedPipeline import GradientBasedPipeline
                return GradientBasedPipeline(self)
            elif self.algorithm == "ANIL":
                from algorithms.ANILPipeline import ANILPipeline
                return ANILPipeline(self)
            else:
                raise ValueError("The given algorithm does not exist. Check if you correctly set mode and lightning usage.")

    def print_config(self):
        ''' Prints the whole training params class.'''
        result_str = "Configurations:\n"
        tab = " "*3

        conf_dict = asdict(self)
        for key, value in conf_dict.items():
            if isinstance(value, dict):
                result_str += f"| {key}:\n"
                for key2, value2 in value.items():
                    if key2 == 'product_list':
                        continue  # Too long for one line for screen reader.
                    result_str += f"| {tab}{key2}: {value2}\n"
            else:
                result_str += f"| {key}: {value}"

        logger.info(result_str)
        print(result_str)                  # ← ADD temporarily


DEFAULT_EVALUATIONS = {
    'adaptation_speed': False,
    'generalization': True,
    'sample_efficiency': False,
    'print_results': True,
}
