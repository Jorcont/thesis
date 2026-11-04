from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Optional, Union, Dict

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.plugins import _PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT
)
from lightning_fabric.utilities.types import _PATH

from conf.ConfigBase import ConfigBase


@dataclass
class TrainerConfig(ConfigBase):
    """ Configuration for creating a PyTorch Lightning ``Trainer``.
    This dataclass mirrors the most relevant arguments of
    ``pytorch_lightning.Trainer`` and is used as the trainer-specific sub-config
    inside the main pipeline configuration. It defines runtime behavior such as
    device usage, distributed strategy, precision, logging, callbacks, validation
    frequency, checkpointing, gradient handling, profiling, and debugging options.

    Some trainer parameters, such as accelerator-related settings, maximum epochs,
    and validation frequency, are initialized or overwritten from the surrounding
    ``PipelineParams`` configuration before the trainer is created.

    Attributes:
        strategy:
            Training strategy to use. Can be ``"auto"``, a strategy name, or a
            PyTorch Lightning ``Strategy`` instance.
        devices:
            Devices to use for training. Can be ``"auto"``, an integer number of
            devices, a list of device indices, or another Lightning-supported device
            specification.
        num_nodes:
            Number of compute nodes used for distributed training.
        precision:
            Numerical precision used during training, for example ``"32-true"``,
            ``"16-mixed"``, or other Lightning-supported precision values.
        logger:
            Logger configuration used by the trainer. Can be a Lightning logger,
            an iterable of loggers, ``True`` for default logging, ``False`` to
            disable logging, or ``None``.
        callbacks:
            List of callbacks or callback identifiers to attach to the trainer.
        fast_dev_run:
            Enables a quick development run. If ``True`` or a positive integer,
            Lightning runs only a small number of batches to check that the
            training, validation, test, and prediction loops work.
        min_epochs:
            Minimum number of training epochs.
        max_steps:
            Maximum number of optimizer steps. A value of ``-1`` means that this
            limit is disabled.
        min_steps:
            Minimum number of optimizer steps.
        max_time:
            Maximum wall-clock training time. Can be a string, a ``timedelta``,
            or a dictionary with time units.
        limit_train_batches:
            Limits the number or fraction of training batches per epoch.
        limit_val_batches:
            Limits the number or fraction of validation batches per validation run.
        limit_test_batches:
            Limits the number or fraction of test batches per test run.
        limit_predict_batches:
            Limits the number or fraction of prediction batches per prediction run.
        overfit_batches:
            Uses a fixed subset of batches for overfitting/debugging. Can be an
            integer number of batches or a fraction.
        val_check_interval:
            Validation interval within an epoch. Can be an integer number of
            training batches or a float fraction of an epoch.
        num_sanity_val_steps:
            Number of validation sanity-check steps to run before training starts.
        log_every_n_steps:
            Logging frequency in training steps.
        enable_checkpointing:
            Whether trainer-level checkpointing is enabled.
        enable_progress_bar:
            Whether the progress bar is enabled.
        enable_model_summary:
            Whether the model summary is printed.
        accumulate_grad_batches:
            Number of batches over which gradients are accumulated before an
            optimizer step is performed.
        gradient_clip_val:
            Gradient clipping value. If ``None``, gradient clipping is disabled.
        gradient_clip_algorithm:
            Gradient clipping algorithm, for example ``"norm"`` or ``"value"``.
        deterministic:
            Whether deterministic operations should be enforced. Can also be set
            to Lightning's warning mode.
        benchmark:
            Whether to enable cuDNN benchmarking for potentially faster training
            with fixed input sizes.
        inference_mode:
            Whether to use ``torch.inference_mode`` during evaluation and prediction.
        use_distributed_sampler:
            Whether Lightning should automatically wrap dataloaders with a
            distributed sampler when using distributed training.
        profiler:
            Profiler configuration. Can be a Lightning profiler instance, a
            profiler name, or ``None``.
        detect_anomaly:
            Enables PyTorch anomaly detection for debugging invalid gradients.
        barebones:
            Runs the trainer with many convenience features disabled for minimal
            overhead and debugging.
        plugins:
            Plugin or list of plugins used by Lightning, for example precision,
            checkpoint I/O, or cluster environment plugins.
        sync_batchnorm:
            Whether to synchronize batch normalization layers across distributed
            processes.
        reload_dataloaders_every_n_epochs:
            Frequency, in epochs, at which dataloaders are reloaded.
        default_root_dir:
            Default directory used by Lightning for logs, checkpoints, and other
            trainer outputs.
        model_registry:
            Optional model registry identifier or path used by the surrounding
            pipeline. This is not a standard Lightning ``Trainer`` argument unless
            explicitly handled elsewhere.

    Notes:
        This class only stores trainer configuration values. The actual
        ``pytorch_lightning.Trainer`` instance is created later by the pipeline
        using these fields.

        Fields that are ``None`` are typically passed through to Lightning, allowing
        the framework to apply its own defaults.
    """
    strategy: Union[str, Strategy] = "auto"
    devices: Union[list[int], str, int] = "auto"
    num_nodes: int = 1
    precision: Optional[_PRECISION_INPUT] = None
    logger: Optional[Union[Logger, Iterable[Logger], bool]] = None
    callbacks: Optional[list[str, Callback]] = field(default_factory=list)
    fast_dev_run: Union[int, bool] = False
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, dict[str, int]]] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_predict_batches: Optional[Union[int, float]] = None
    overfit_batches: Union[int, float] = 0.0
    val_check_interval: Optional[Union[int, float]] = None
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None
    deterministic: Optional[Union[bool, _LITERAL_WARN]] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = False
    use_distributed_sampler: bool = True
    profiler: Optional[Union[Profiler, str]] = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[_PATH] = None
    model_registry: Optional[str] = None
    accelerator:             str   = "auto"   

    def create_checkpoint_callback(self, checkpoint_dir, keep_checkpoints, checkpoint_freq, patience=10):
        ''' Creates customized model callback for custom checkpointing.
        Note: The meta_loss is the test loss achieved during validation averaged over all validation tasks.'''
        from evaluation.memory_callback import MemoryTrackerCallback 
        
        self.callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=keep_checkpoints,
            every_n_epochs=checkpoint_freq,
            monitor= "meta_val_loss",
            mode="min"
        ))
        self.callbacks.append(MemoryTrackerCallback())
