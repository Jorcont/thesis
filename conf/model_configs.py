''' Configs for model creation.'''

import logging
import pathlib
from dataclasses import dataclass, field, InitVar
from typing import Literal, Optional, Dict, Any, List
import warnings
import torch

from conf.data_configs import DataConfig
from conf.loss_configs import LossConfig
from conf.sampler_configs import SamplerConfig
from models.ppnet import BackboneClassifier
from utils.path_utils import template_to_path
from algorithms.PipelineState import PipelineState
from pathlib import Path

logger = logging.getLogger("meta-netze")

@dataclass
class ModelConfig:
    """ Base class for model creation.
    Whenever a new model shell be used, add the corresponding config here by
    1. Writing the corresponding sub-config as one of the classes below.
    2. Add the model name to the possible names of this base class.
    3. Write the corresponding if statement in the post_init of this base class.

    Attributes:
        name: Defines which model to use.
        config: Extra information for model creation like input and output sizes.
            Note: config entries can also be used for recursively defining model components.
        checkpoint: Which checkpoint to load for the model.
            If None, no checkpoint is loaded and model is randomly initialized.
            Note: This only works for public checkpoints from the web.
            For loading local checkpoints, please use the PipelineConfig in configs.py.
    Note that this setup enables loading checkpoints for only model blocks/parts, e.g., the backbone of a large-scale architecture.
    """
    device: torch.device
    name: Literal["TFT", "PEARLSAC", "Regressor", "DiscretePolicy", "GausianContextPolicy",
    "ContextQFunction", "ProtoPNet", "ConvNext", "LSTMRegressor",
    "TFTRegressor", "TCNRegressor", "PF_TFT","GaussianContextPolicy", "ContextEncoder", "ContextQFunction", "ProtoPNet", "ConvNext"] = None
    config: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)


    def __post_init__(self) -> None:
        if self.name == "TFT":
            from pytorch_forecasting import TemporalFusionTransformer
            self.config = TFTConfig(**self.config).__dict__
            self.model = TemporalFusionTransformer.from_dataset(**self.config)
            self.model.to(self.device)
        elif self.name == "PEARLSAC":
            from models.Agents import PEARLSACAgent
            self.config = PEARLSACConfig(**self.config).__dict__
            self.model = PEARLSACAgent(**self.config)
        elif self.name == "ProtoPNet":
            from models.ppnet import PPNet
            self.config = ProtoPNetConfig(**self.config).__dict__
            self.model = PPNet(**self.config)
        elif self.name == "BackboneClassifier":
            from models.ppnet import BackboneClassifier
            self.config = BackboneClassifierConfig(**self.config).__dict__
            self.model = BackboneClassifier(**self.config)
        elif self.name == 'DiscretePolicy':
            from models.Policys import DiscretePolicy
            module_config = SimpleModule(**self.config)
            self.config = module_config.__dict__
            self.model = DiscretePolicy(
                obs_dim=module_config.in_dim,
                act_dim=module_config.out_dim,
                hidden_dims=module_config.hidden_dims,
            )
        elif self.name == "ContextEncoder":
            from models.Encoders import ContextEncoder
            module_config = ContextRLModule(**self.config)
            self.config = module_config.__dict__
            self.model = ContextEncoder(
                obs_dim=module_config.in_dim,
                act_dim=module_config.out_dim,
                z_dim=module_config.z_dim,
                hidden_dims=module_config.hidden_dims,
            )
        elif self.name == "ConvNext":
            from models.convnext import ConvNextClassifier
            self.config = ConvNextConfig(**self.config).__dict__
            self.model = ConvNextClassifier(**self.config)
        elif self.name == "SmallCNN":
            from models.CNN import CNN
            pool = self.config.get("pool")
            self.model = CNN(pool)
        elif self.name == "ContextQFunction":
            from models.Critics import ContextQFunction
            module_config = ContextRLModule(**self.config)
            self.config = module_config.__dict__
            self.model = ContextQFunction(
                obs_dim=module_config.in_dim,
                act_dim=module_config.out_dim,
                z_dim=module_config.z_dim,
                hidden_dims=module_config.hidden_dims,
            )
        elif self.name in {'GausianContextPolicy', 'GaussianContextPolicy'}:
            from models.Policys import GaussianContextPolicy
            module_config = ContextRLModule(**self.config)
            self.config = module_config.__dict__
            self.model = GaussianContextPolicy(
                obs_dim=module_config.in_dim,
                act_dim=module_config.out_dim,
                z_dim=module_config.z_dim,
                hidden_dims=module_config.hidden_dims,
            )
        elif self.name == 'Regressor':
            from models.RegressionModels import Regressor
            self.config = SimpleModule(**self.config).__dict__
            self.model = Regressor(**self.config)
        elif self.name == "LSTMRegressor":
            from models.RegressionModels import LSTMRegressor
            from dataclasses import asdict
            lstm_cfg = LSTMRegressorConfig(**self.config)
            self.config = asdict(lstm_cfg)
            self.model = LSTMRegressor(**self.config)
        elif self.name == "TFTRegressor":
            from models.RegressionModels import TFTRegressor
            from dataclasses import asdict
            tft_cfg = TFTRegressorConfig(**self.config)
            self.config = asdict(tft_cfg)
            self.model = TFTRegressor(self.config)
        elif self.name == "TCNRegressor":
            from models.RegressionModels import TCNRegressor
            from dataclasses import asdict
            tcn_cfg     = TCNRegressorConfig(**self.config)
            self.config = asdict(tcn_cfg)
            self.model  = TCNRegressor(self.config)
        elif self.name == "PF_TFT":
            from models.RegressionModels import PlainTFT
            from data.pf_tft_utils import create_minimal_pf_dataset, verify_pf_dict_format
            from pytorch_forecasting import TemporalFusionTransformer
            from dataclasses import asdict
            pf_cfg = PF_TFTConfig(**self.config)
            self.config = asdict(pf_cfg)

            # Create minimal dataset for architecture configuration
            dataset = create_minimal_pf_dataset(
                time_varying_known_reals=pf_cfg.time_varying_known_reals,
                time_varying_unknown_reals=pf_cfg.time_varying_unknown_reals,
                target=pf_cfg.target,
                max_encoder_length=pf_cfg.max_encoder_length,
                max_prediction_length=pf_cfg.max_prediction_length,
                static_reals = pf_cfg.static_reals,
            )
            # Instantiate TFT from dataset metadata
            tft = TemporalFusionTransformer.from_dataset(
                dataset,
                hidden_size=pf_cfg.hidden_size,
                attention_head_size=pf_cfg.attention_head_size,
                dropout=pf_cfg.dropout,
                hidden_continuous_size=pf_cfg.hidden_continuous_size,
                output_size=pf_cfg.output_size,
                learning_rate=pf_cfg.learning_rate,
                log_interval=-1)
            # Wrap in PlainTFT for pipeline compatibility
            self.model = PlainTFT(tft)
            print(f"[PF_TFT] Model created — "
                  f"params: {sum(p.numel() for p in self.model.parameters()):,}")
        else:
            raise ValueError(f"Unknown model name: {self.name}")
    
        if self.checkpoint:
            logger.info("Starting training from public checkpoint.")
            self.checkpoint: CheckpointConfig = CheckpointConfig(**self.checkpoint)
            self.load_checkpoint()

    def load_checkpoint(self):
        ''' Loads the checkpoint specified in the self.checkpoint dict
        into self.model.'''
        # Only huggingface checkpoints possible yet.
        self.load_hf_checkpoint()

    def load_hf_checkpoint(self):
        """
        Load a pretrained Hugging Face model + tokenizer.
        Example: model_name = "facebook/convnext-base-224-22k"
        """
        from transformers import AutoModel, AutoTokenizer
        logger.info("Loading Hugging Face checkpoint from the web.")
        self.model = AutoModel.from_pretrained(self.checkpoint.path)
        logger.info("Finished loading model.")

@dataclass
class TFTConfig:
    ''' Configuration for the Temporal Fusion Transformer.'''
    # Todo: Those are not the right types.
    log_interval: int = 1
    dropout: float = 0.1
    # Todo: This should be the meta_lr or the inner_lr, depending on the used algorithm.
    learning_rate: float = 0.03
    attention_head_size: int = 1
    hidden_continuous_size: int = 8# set to <= hidden_size
    hidden_size: int = 8

    def __post_init__(self) -> None:
        sampler = SamplerConfig.get_config().sampler
        sampler.train_dataset.sample_task()
        self.dataset  = sampler.train_dataset.data
        self.loss = LossConfig.get_config().loss_fn.basic_loss_fn

@dataclass
class PEARLSACConfig:
    ''' Specific config for the PEARLSACAgent class.
    Itself contains several other models that together build the Agent.

    Args:
        - context_encoder_config: Contains name and config of context encoder z.
        policy_config: Contains name and config for the policy \pi.
        critic_config: Contains name and config for the critics Q1, Q2, target_Q1  and target_Q2.
    '''
    policy_config: dict = field(default_factory=dict)
    context_encoder_config: dict = field(default_factory=dict)
    critic_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ''' Assigns nested config dicts to the different model classes.'''
        self.policy = ModelConfig(**self.policy_config).model
        self.context_encoder = ModelConfig(**self.context_encoder_config).model
        # All critics of SAC.
        self.Q1 = ModelConfig(**self.critic_config).model
        self.Q2 = ModelConfig(**self.critic_config).model
        self.target_Q1 = ModelConfig(**self.critic_config).model
        self.target_Q2 = ModelConfig(**self.critic_config).model

@dataclass
class ProtoPNetConfig:
    ''' Specific config for the PPNet class.
    Itself contains other models that together build the Agent.

    Args:
        - backbone_config: Contains name and config of backbone model.
                    prototype_shape (Tuple[int, int, int, int]): A tuple representing the shape of the prototypes in the PPNet,
                as (number of prototypes, number of channels, prototype height, prototype width).
            topk_k (int): The number of top prototype activations to consider during training (default=1).
                i.e. defaults to max pooling where only the maximum patch activation is considered.
            margin (Optional[float]): The margin to use for subtractive margin cross entropy. (default=None).
            init_weights (bool): A boolean indicating whether to initialize the weights of the PPNet (default=True).
            add_on_layers_type (str): A string indicating the type of additional layers to add to the base architecture,
                can be 'bottleneck', 'identity', or 'upsample' (default='bottleneck').
            incorrect_class_connection (float): A float value incorrect class connections are initialized to
    '''
    prototype_shape: Dict[str, int]
    backbone_config: dict = field(default_factory=dict)
    topk_k: int = 1
    margin: Optional[float] = None
    init_weights: bool = True
    add_on_layers_type: Literal["bottleneck", "identity", "upsample"] = "identity"
    weight_decay: float = 0.1
    incorrect_class_connection: Optional[float] = -0.5
    correct_class_connection: float = 1.0
    bias_last_layer: Optional[float] = None
    non_negative_last_layer: bool = True
    embedded_spectrogram_height: Optional[int] = None
    pretrain_info: Optional[Dict] = None
    use_as_backbone: bool = False
    shared_prototypes: bool = False

    def __post_init__(self) -> None:
        """ Finalize the ProtoPNet configuration.
        This method derives additional PPNet settings from the global data and backbone
        configurations. It determines the number of classes from the active
        ``DataConfig``, injects PPNet-specific values into the nested backbone
        configuration, creates the configured backbone model, and validates that the
        backbone configuration provides the embedding size required by PPNet.

        Side Effects:
            - Adds ``num_classes`` to this config instance based on
            ``DataConfig.get_config().config.m_way``.
            - Updates ``backbone_config["config"]`` with:
                - ``num_classes`` set to the detected number of classes.
                - ``backbone_mode`` set to ``True``.
            - Instantiates ``ModelConfig`` from ``backbone_config``.

        Raises:
            KeyError:
                If ``backbone_config`` does not contain the expected nested
                ``"config"`` dictionary.
            AttributeError:
                If no global ``DataConfig`` is available, or if it does not provide
                ``config.m_way``.
            TypeError:
                If ``backbone_config`` contains invalid arguments for ``ModelConfig``.
            ValueError:
                If the resolved backbone config does not define ``embedding_size``,
                which is required for PPNet construction.

        Notes:
            This method expects that a global ``DataConfig`` has already been created
            before ``ProtoPNetConfig`` is initialized.
            This should, however, be the case when using the pre-defined shape of configs.py

            The backbone is created in ``backbone_mode`` so that it can be used as the
            feature extractor inside the PPNet architecture.
        """
        self.num_classes = DataConfig.get_config().config.m_way
        self.embedding_size = self.prototype_shape["num_prototypes"]

        if self.num_classes is None:
            # Arbitrary set to 3. The output layer will be set in each model run w.r.t. the obtained batch.
            self.num_classes = 3
            logging.info("Set number of classes to 3, since no M_way attribute found. It is expected, that the model's last layer is set in the module for each batch.")
        self.backbone_config["config"]['num_classes'] = self.num_classes
        self.backbone_config["config"]['backbone_mode'] = True

        backbone = ModelConfig(**self.backbone_config)
        self.backbone_model = backbone.model

        if "embedding_size" not in backbone.config:
            raise ValueError("Backbone config must contain embedding size for PPNet.")

@dataclass
class BackboneClassifierConfig:
    ''' Specific config for the BackboneClassifier.'''
    backbone_config: dict = field(default_factory=dict)
    embedding_size: int = 128

    def __post_init__(self) -> None:
        self.num_classes = DataConfig.get_config().config.m_way
        self.backbone_config["config"]['num_classes'] = self.num_classes
        self.backbone_config["config"]['backbone_mode'] = True

        backbone = ModelConfig(**self.backbone_config)
        self.backbone_model = backbone.model
        self.in_features = self.embedding_size

        if "embedding_size" not in backbone.config:
            raise ValueError("Backbone config must contain embedding size for PPNet.")

@dataclass
class ConvNextConfig:
    ''' Specific config for the ConvNext model.
    It is used as a backbone for PPNet, but can also be used individually.

    Args:
        num_classes (int): The number of classes for classification.
        num_channels (int): The number of input channels. Default is 1. Set to 3 for rgb images.
        embedding_size: The size of the embeddings, the model will produce as output.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights. Default is None.
        - local_checkpoint: If a local checkpoint should be used instead of loading one from the net.
    '''
    num_classes: int = None
    num_channels: int = 1
    embedding_size: int = 1024
    backbone_mode: bool = True
    checkpoint: Optional[str] = None
    local_checkpoint: Optional[str] = None
    cache_dir: Optional[str] = None
    pretrain_info: Optional[Dict] = None

@dataclass
class SimpleModule:
    ''' Configures a very simple discrete module, e.g. a policy.
    '''
    in_dim: int = None
    out_dim: int = None
    hidden_dims: tuple = (256,256)

    def __post_init__(self) -> None:
        if isinstance(self.hidden_dims, list):
            self.hidden_dims = tuple(self.hidden_dims)
        if self.in_dim is None:
            raise ValueError("No input dimension provided")
        elif self.out_dim is None:
            raise ValueError("No output dimension provided.")

@dataclass
class ContextRLModule:
    ''' Configures a RLModule, e.g. a policy, that receives a context encoding as input.
    '''
    # Todo: Here, still need to replace.
    in_dim: int = None
    out_dim: int = None
    hidden_dims: tuple = (256,256) # Todo: This doesn't work, I guess.
    z_dim: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.hidden_dims, list):
            self.hidden_dims = tuple(self.hidden_dims)
        if self.in_dim is None:
            raise ValueError("No input dimension provided")
        elif self.out_dim is None:
            raise ValueError("No output dimension provided.")

# Todo: Who has created this and what for?
@dataclass
class CheckpointConfig:
    path: str = None
    device: str = "cpu"

    def __post_init__(self):
        if self.path is None:
            raise ValueError("You must provide a path for your checkpoint.")

        self.path: Path = template_to_path(self.path)

@dataclass
class LSTMRegressorConfig:
    """Configuration for LSTMRegressor."""
    input_size:  int   = 3
    hidden_size: int   = 64
    num_layers:  int   = 2
    output_size: int   = 1
    look_ahead:  int   = 96
    dropout:     float = 0.2

    def __post_init__(self):
        if self.input_size is None:
            raise ValueError("input_size must be provided.")


@dataclass
class TFTRegressorConfig:
    """
    Configuration for the MAML-compatible TFTRegressor.

    enc_size    : number of encoder input features
                  = len(time_varying_known_reals)
                  + len(time_varying_unknown_reals excl. target)
    dec_size    : number of decoder input features
                  = len(time_varying_known_reals) only
                  (future timesteps: only known variables available)
    hidden_size : internal TFT hidden dimension
    num_heads   : number of attention heads (must divide hidden_size)
    num_layers  : number of LSTM encoder layers
    dropout     : dropout rate
    look_ahead  : forecast horizon in steps  (= max_prediction_length)
    output_size : number of target variables (1 for single-target)
    """
    # Just write them, in the case some key is missing in the YAML file.
    # YAML can always override, but this can sometimes be useful for debugging.
    enc_size:    int   = 9
    dec_size:    int   = 9
    hidden_size: int   = 64
    num_heads:   int   = 4 # number of attention heads!!
    num_layers:  int   = 2
    dropout:     float = 0.1
    look_ahead:  int   = 96
    output_size: int   = 1

    def __post_init__(self):
        # Raise very important error!!
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible "
            f"by num_heads ({self.num_heads})"
        )

@dataclass
class TCNRegressorConfig:
    """
    Configuration for the non-causal TCNRegressor.

    Receptive field (non-causal, symmetric dilation):
        RF = 1 + 2 × (kernel_size - 1) × Σ(2^i, i=0..n_blocks-1)
        Default (k=3, n=6): RF = 253 steps  >>  look_back=96  ✓
    """
    input_size:  int   = 13    # len(PVTimeSeriesDatasetV2.FEATURE_COLS)
    n_filters:   int   = 64    # channel width (body depth)
    kernel_size: int   = 3     # must be odd — symmetric non-causal padding
    n_blocks:    int   = 6     # dilation schedule: 1, 2, 4, 8, 16, 32
    dropout:     float = 0.1   # Dropout (paired with LayerNorm)
    look_back:   int   = 96    # encoder window  (24h @ 15-min)
    look_ahead:  int   = 96    # forecast horizon (24h @ 15-min)
    output_size: int   = 1     # single-target point forecast

    def __post_init__(self) -> None:
        assert self.kernel_size % 2 == 1, \
            "kernel_size must be odd for exact symmetric non-causal padding."
        dilations = [2 ** i for i in range(self.n_blocks)]
        rf = 1 + 2 * (self.kernel_size - 1) * sum(dilations)
        n_body_params = (
            self.input_size * self.n_filters                        # input_proj
            + self.n_blocks * 2 * self.n_filters ** 2 * self.kernel_size  # blocks
        )
        if rf < self.look_back:
            import warnings
            warnings.warn(
                f"[TCNRegressorConfig] Receptive field ({rf} steps) < "
                f"look_back ({self.look_back} steps). "
                f"Increase n_blocks or kernel_size."
            )
        print(
            f"[TCNRegressorConfig] "
            f"RF={rf} steps | look_back={self.look_back} | "
            f"body≈{n_body_params:,} params | "
            f"dropout={self.dropout}"
        )

@dataclass
class PF_TFTConfig:
    """
    Configuration for pytorch_forecasting TemporalFusionTransformer.

    The model is instantiated via from_dataset() using a minimal TimeSeriesDataSet.
    These parameters control the TFT architecture.
    """
    # ── Architecture ──────────────────────────────────────────────
    hidden_size:            int   = 32
    attention_head_size:    int   = 2
    dropout:                float = 0.2
    hidden_continuous_size: int   = 16
    output_size:            int   = 7      # number of quantiles (or 1 for point forecast)
    learning_rate:          float = 0.001  # required by from_dataset, overridden by meta_lr

    # ── Data schema (must match PVTimeSeriesDatasetV2 + YAML) ─────
    max_encoder_length:  int = 96
    max_prediction_length: int = 24
    target: str = "active_power_kW"

    time_varying_known_reals: List[str] = field(default_factory=lambda: [
        "WIND_DEGREES_40M",
        "PRESSURE_SURFACE",
        "DIRECT_HORIZONTAL_IRRADIANCE",
        "WIND_SPEED_40M",
        "TEMPERATURE_2METER_ABOVE_GROUND",
        "HUMIDITY_RELATIVE",
        "DIFFUSE_HORIZONTAL_IRRADIANCE",
        "SOLAR_ELEVATION",
        "SOLAR_AZIMUTH",
        "TIME_OF_DAY_SIN",
        "TIME_OF_DAY_COS",
        "DAY_OF_YEAR_SIN",
        "DAY_OF_YEAR_COS",
    ])

    time_varying_unknown_reals: List[str] = field(default_factory=lambda: [
        "active_power_kW",
    ])
    static_reals: List[str] = field(default_factory=lambda: [
    "LATITUDE",
    "LONGITUDE"])
    n_static_reals: int = 2

    def __post_init__(self):
        assert self.hidden_size % self.attention_head_size == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"attention_head_size ({self.attention_head_size})"
        )
