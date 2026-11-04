from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from conf.ConfigBase import ConfigBase
from losses.BasicLoss import BasicLoss
from losses.TimeSeriesLoss import TimeSeriesLoss


@dataclass
class LossConfig(ConfigBase):
    """ Base class for loss functions.
    Similar to all other sub-configs, one can add losses here, by
    1. writing a corresponding class below.
    2. Adding the class name to the names possible for this base class.
    3. Writing the corresponding if statement in the post_init.

        Attributes:
            name (optional): Sets the name of the loss type. Defaults to MAE.
            config (optional): A dictionary of arguments to be passed to the loss function constructor.
    """
    name: Literal["L1", "NLL", "MSE", "PrototypeLoss", "CrossEntropy", "BCE", "MAE", "Quantile"] = "L1"
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name == "L1":
            self.config = L1LossConfig(**self.config).__dict__
        elif self.name == "NLL":
            self.config = NLLLossConfig(**self.config).__dict__
        elif self.name == "MSE":
            self.config = MSELossConfig(**self.config).__dict__
        elif self.name == "CrossEntropy":
            self.config = CrossEntropyLossConfig(**self.config).__dict__
        elif self.name == "PrototypeLoss":
            self.config = PrototypeLossConfig(**self.config).__dict__
        elif self.name == "LinearClassifier":
            self.config = LinearClassifierConfig(**self.config).__dict__
        elif self.name == "BCE":
            self.config = BCELossConfig(**self.config).__dict__
        elif self.name == "TFT-MAE":
            self.config = MAEConfig(**self.config).__dict__
        elif self.name == "Quantile":
            from losses.quantile_loss import (QuantileLoss, PVRegressionLoss)
            from dataclasses import asdict
            q_cfg              = QuantileLossConfig(**self.config)
            ql                 = QuantileLoss(q_cfg.quantiles, q_cfg.median_only)
            self.config        = asdict(q_cfg)
            self.config["loss_fn"] = PVRegressionLoss(basic_loss_fn=ql)
        elif self.name == "TCNMSELoss":
            self.config = TCNMSELossConfig(**self.config).__dict__
        elif self.name == "TCNHuberLoss":
            self.config = TCNHuberLossConfig(**self.config).__dict__
        else:
            raise ValueError(f"Unknown loss function: {self.name}")

        # ── Set self.loss_fn ──────────────────────────────────────────
        raw = self.config.get('loss_fn')
        if raw is None:
            raise ValueError("loss_fn is not set correctly.")

        if self.name == "TFT-MAE":
            self.loss_fn = TimeSeriesLoss(raw)
        elif hasattr(raw, 'calculate_loss'):
            # Already PVRegressionLoss — use as-is
            self.loss_fn = raw
        elif self.name in ("BCE", "L1", "NLL"):
            # Keep original BasicLoss for non-regression tasks
            self.loss_fn = BasicLoss(raw)
        elif self.name in ("TCNMSELoss", "TCNHuberLoss"):
            self.loss_fn = raw
        else:
            # MSE, CrossEntropy etc. → wrap in PVRegressionLoss
            from losses.quantile_loss import PVRegressionLoss
            self.loss_fn = PVRegressionLoss(basic_loss_fn=raw)

@dataclass
class L1LossConfig:
    """ Configuration for L1 Loss (Mean Absolute Error). """
    size_average: bool = True
    reduce: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        self.loss_fn = nn.L1Loss(
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction
        )

@dataclass
class NLLLossConfig:
    """ Configuration for Negative Log Likelihood Loss. """
    weight: Optional[List[float]] = None
    size_average: bool = True
    ignore_index: Optional[int] = None
    reduce: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        weight_tensor = torch.tensor(self.weight) if self.weight else None
        self.loss_fn = nn.NLLLoss(
            weight=weight_tensor,
            size_average=self.size_average,
            ignore_index=self.ignore_index,
            reduce=self.reduce,
            reduction=self.reduction
        )

@dataclass
class MSELossConfig:
    """ Configuration for Mean Squared Error Loss. """
    size_average : bool = True
    reduce : bool = True
    reduction : Literal["none", "mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        self.loss_fn = nn.MSELoss(
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction
        )

@dataclass
class CrossEntropyLossConfig:
    """ Configuration for Cross Entropy Loss. """
    weight: Optional[List[float]] = None
    size_average: bool = True
    ignore_index: int = -1
    reduce: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        weight_tensor = torch.tensor(self.weight) if self.weight else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight_tensor,
            size_average=self.size_average,
            ignore_index=self.ignore_index,
            reduce=self.reduce,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )

@dataclass
class PrototypeLossConfig:
    """ Configuration for prototype larning with Cross Entropy and orthogonality Loss. """
    basic_loss_config: dict = field(default_factory=dict)
    coefficients: dict = field(default_factory=dict)
    higher: bool = False
    orthogonality: bool = True

    def __post_init__(self) -> None:
        from losses.prototype_losses import PrototypeLoss

        # Todo: Recursive loss setting does not work anymore as global config overwrits itself.
        self.basic_loss = CrossEntropyLossConfig(**self.basic_loss_config['config']).loss_fn
        self.loss_fn = PrototypeLoss(basic_loss_fn=self.basic_loss, coefficients=self.coefficients, higher =self.higher, orthogonality = self.orthogonality)

@dataclass
class LinearClassifierConfig:
    """ Configuration for prototype larning with Cross Entropy and orthogonality Loss. """
    basic_loss_config: dict = field(default_factory=dict)
    coefficients: dict = field(default_factory=dict)
    higher: bool = False


    def __post_init__(self) -> None:

        # Todo: Recursive loss setting does not work anymore as global config overwrits itself.
        self.basic_loss = CrossEntropyLossConfig(**self.basic_loss_config['config']).loss_fn
        from losses.BasicLoss import BasicLoss
        self.loss_fn = BasicLoss(self.basic_loss, self.higher)

@dataclass
class BCELossConfig:
    """ Configuration for Binary Cross Entropy Loss. """
    weight: Optional[List[float]] = None
    size_average: bool = True
    reduce: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        weight_tensor = torch.tensor(self.weight) if self.weight else None
        self.loss_fn = nn.BCELoss(
            weight=weight_tensor,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction
        )

@dataclass
class MAEConfig:
    ''' Mean absolute error loss configs.'''
    reduction: str = 'mean'
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from pytorch_forecasting.metrics import MAE
        self.loss_fn = MAE(reduction=self.reduction, **self.kwargs)

@dataclass
class QuantileLossConfig:
    """
    quantiles   : list of quantile levels to predict -> YAML
    median_only : if True, only the median (q=0.5) pinball loss is computed
                  → used during MAML inner-loop adaptation steps for speed
                    and gradient stability
    """
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    median_only: bool = False

    def __post_init__(self):
        assert 0.5 in self.quantiles, "Quantiles must include 0.5 (median)."

class TCNMSELoss:
    """
    Standard MSE loss for TCNRegressor.
    Interface: (model, X, y) → (scalar_loss, predictions)

        X : [B, look_back, input_size]   plain tensor — LSTM mode
        y : [B, look_ahead, output_size]
    """
    def __call__(self,
        model: torch.nn.Module,
        X:     torch.Tensor,
        y:     torch.Tensor,
    ):
        if isinstance(X, list):
            X = torch.stack(X) # [k, look_back, n_features]
        preds = model(X)               # [B, look_ahead, output_size]
        loss  = F.mse_loss(preds, y)   # scalar
        return loss, preds

class TCNHuberLoss:
    """
    Huber loss for TCNRegressor — RECOMMENDED for PV forecasting.
    Why Huber over MSE for PV:
      • PV output = 0 for ~60-70% of timesteps (night) → many small errors
      • MSE over-penalises occasional large peak-hour errors
      • Huber: MAE for |error|>delta (robust to spikes),
               MSE for |error|<delta (stable near convergence)
      • delta=1.0 good default; scale to ~5% of max_power_kW if needed
    Interface: (model, X, y) → (scalar_loss, predictions)
    """
    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def __call__(self,
        model: torch.nn.Module,
        X:     torch.Tensor,
        y:     torch.Tensor,
    ):
        if isinstance(X, list):
            X = torch.stack(X)   # [k, look_back, n_features]
        preds = model(X) # compute the predictions
        loss  = F.huber_loss(preds, y, delta=self.delta)
        return loss, preds
    
@dataclass
class TCNMSELossConfig:
    def __post_init__(self) -> None:
        self.loss_fn = TCNMSELoss()

@dataclass
class TCNHuberLossConfig:
    delta: float = 1.0
    def __post_init__(self) -> None:
        self.loss_fn = TCNHuberLoss(delta=self.delta)