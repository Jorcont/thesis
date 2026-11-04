""" The main loss for Prototype-Learning."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

class PrototypeLoss:
    ''' Bundles functions for calculating loss for ProtoPNet.'''
    def __init__(self, basic_loss_fn,     coefficients: Dict = None, higher: bool = False, orthogonality: bool = True):
        ''' This class gets initialized during configuration.

        Args:
            basic_loss_fn: The basic loss function, e.g., cross-entropy.
            coefficients : Dict of coefficients . If none, default values are used.
        '''
        print('Init Prototype_Loss directly')
        self.higher = higher
        self.orthogonality = orthogonality
        print("HIGHER: ", self.higher)
        self.basic_loss_fn = basic_loss_fn
        self.default_coeffs = {
            "basic_loss": 1.0,
            "orthogonality_loss": 1.0,
        }
        if coefficients is not None:
            # Todo: does this work?
            self.default_coeffs.update(coefficients)

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        ''' Unwrape model if multi-GPU usage.'''
        return model.module if hasattr(model, "module") else model

    def calculate_orthogonality_loss(self, model) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        if not self.higher:
            model = self._unwrap(model)
        orthogonalities = model.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)
        return orthogonality_loss / orthogonalities.numel()

    def calculate_total_loss(self, model, X, y) -> torch.Tensor:
        """
        Compute the total loss by combining orthogonality loss with the given loss_fn.

        Args:
            model: model for forward path.
            X: The input to be processed by model.
            y: True labels.

        Returns:
            torch.Tensor: The computed total loss.
            preds: The logits leading to the loss. (Those are just for metric tracking purposes).
        """
        if not self.higher:
            model = self._unwrap(model)  # <-- WICHTIG: auch hier ent-wrappen
        # Forward flow through the ProtoPnet.
        logits, _ = model(X, prototypes_of_wrong_class=None)

        basic_loss = self.default_coeffs["basic_loss"] * self.basic_loss_fn(logits, y)
        if self.orthogonality:
            orthogonality_loss = self.calculate_orthogonality_loss(model) * self.default_coeffs["orthogonality_loss"]
        else:
            orthogonality_loss = 0

        return basic_loss + orthogonality_loss, torch.argmax(logits, dim=-1)

    def __call__(self, model, X, y):
        return self.calculate_total_loss(model, X, y)