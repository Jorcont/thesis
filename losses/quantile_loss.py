
import torch
import torch.nn as nn
from typing import List, Optional
from losses.BasicLoss import BasicLoss
import os

class QuantileLoss(nn.Module):
    """
    Pinball loss with peak weighting.
    Peak weighting — high target values get more weight.

    Inputs
    ------
    pred   : [B, look_ahead, n_quantiles]
    target : [B, look_ahead] or [B, look_ahead, 1]
    """

    def __init__(
        self,
        quantiles:  List[float] = None,
        median_only: bool       = False,
        peak_alpha:  float      = 3.5,    # 0.0 = no peak focus
    ):
        super().__init__()
        self.quantiles   = quantiles or [0.1, 0.5, 0.9]
        self.median_only = median_only
        self.median_idx  = self.quantiles.index(0.5)
        self.peak_alpha  = peak_alpha

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if target.dim() == 3:
            target = target.squeeze(-1)            # [B, look_ahead]

        # Peak weight: 1.0 at zero, (1 + alpha) at peak=1.0
        peak_weight = 1.0 + self.peak_alpha * target.detach()

        # ── Median-only (inner-loop fast path) ────────────────
        if self.median_only:
            p   = pred[..., self.median_idx]       # [B, look_ahead]
            err = torch.abs(p - target) * peak_weight
            return err.mean()

        # ── Full pinball loss ─────────────────────────────────
        losses = []
        for i, q in enumerate(self.quantiles):
            p       = pred[..., i]
            err     = target - p
            pinball = torch.max(q * err, (q - 1) * err)   # [B, look_ahead]
            losses.append(pinball * peak_weight)

        return torch.stack(losses).mean()

# ======================================================================= #

class PVRegressionLoss(BasicLoss):
    """
    Regression loss wrapper for LSTM and TFTRegressor.

    Flow:
      1. Forward pass   → raw predictions
      2. Clamp          → max=1.0
      3. Sort quantiles → monotonicity
      4. Save pred_raw  → for night loss gradient
      5. Mask preds     → night output = 0.0
      6. Compute loss   → day pinball + night MSE
    """

    def calculate_loss(
        self,
        model,
        X,
        y,
        night_threshold: float = 1e-3,
    ) -> tuple:

        model = self._unwrap(model)

        # ── Forward pass ─────────────────────────────────────────────
        if isinstance(X, dict):
            preds = model(X)

        elif isinstance(X, tuple) and len(X) == 2 \
                and isinstance(X[0], torch.Tensor) and X[0].dim() == 3:
            preds = model(X[0], X[1])

        elif isinstance(X, tuple) and len(X) == 2 \
                and isinstance(X[0], torch.Tensor) and X[0].dim() == 2:
            preds = model(X)

        elif isinstance(X, torch.Tensor) and X.dim() == 3:
            preds = model(X)

        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X = {
                k: torch.stack([d[k] for d in X])
                   if isinstance(X[0][k], torch.Tensor) else X[0][k]
                for k in X[0].keys()
            }
            preds = model(X)

        elif isinstance(X, list) and len(X) == 2 \
                and isinstance(X[0], torch.Tensor) and isinstance(X[1], torch.Tensor) \
                and X[0].shape != X[1].shape and X[0].dim() == 3:
            preds = model(X[0], X[1])

        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], torch.Tensor):
            preds = model(torch.stack(X))

        else:
            preds = model(X)

        # ── Clamp + Monotonicity ──────────────────────────────────────
        preds = torch.clamp(preds, max=1.0)

        if preds.shape[-1] > 1:
            preds, _ = torch.sort(preds, dim=-1)    # q0.1 ≤ q0.5 ≤ q0.9

        # Save RAW preds → night loss can penalize non-zero predictions
        pred_raw = preds.clone()

        # Zero out night predictions → output always 0 at night
        # preds_out = preds * daytime_mask.unsqueeze(-1)         # [B, look_ahead, Q]
        preds_out = preds
        # ── Combined Loss ─────────────────────────────────────────────
        # day_loss  : pinball + peak_weight, mean over daytime steps only
        # night_loss: MSE on raw median pred at night → model learns 0
        loss = self.basic_loss_fn(
            preds_out,
            y,
            # daytime_mask = daytime_mask,
            # pred_raw     = pred_raw,
        )
        return loss, preds_out