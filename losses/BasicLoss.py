""" The main loss for Prototype-Learning."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

class BasicLoss:
    ''' Wrapper for any default loss function.'''
    def __init__(self, basic_loss_fn, higher:bool=False):
        self.basic_loss_fn = basic_loss_fn
        self.higher = higher

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        ''' Unwrape model if multi-GPU usage.'''
        return model.module if hasattr(model, "module") else model

    def calculate_loss(self, model, X, y) -> torch.Tensor:
        """ Orchestrates computing the moss with given model and data.
        Results in all the different substeps like logits calculation.
        Returns: The calculated error and the prediction made by the model.
        """
        if not self.higher:
            model = self._unwrap(model)  # <-- WICHTIG: auch hier ent-wrappen
        # Todo: Not sure whether the following is required.
        #device = next(self.model.parameters()).device
        #if isinstance(X, torch.Tensor) and X.device != device:
         #   X = X.to(device)
        #if isinstance(y, torch.Tensor) and y.device != device:
         #   y = y.to(device)
        logits = model(X)
        print(f"logits: {logits.shape} | y: {y.shape} | "
          f"logits_range: [{logits.min():.3f}, {logits.max():.3f}] | "
          f"y_range: [{y.min():.3f}, {y.max():.3f}]")
        loss = self.basic_loss_fn(logits, y)
        # preds = torch.argmax(logits, dim=-1)
        preds  = logits.detach()

        return loss, preds

    def __call__(self, model, X, y):
        return self.calculate_loss(model, X, y)