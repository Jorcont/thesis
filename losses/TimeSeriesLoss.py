""" The main loss for Prototype-Learning."""

import torch

from losses.BasicLoss import BasicLoss

class TimeSeriesLoss(BasicLoss):
    def calculate_loss(self, model, X, y) -> torch.Tensor:
        """ Just different model usage here.
        """
        model = self._unwrap(model)
        # Todo: Not sure whether the following is required.
        #device = next(self.model.parameters()).device
        #if isinstance(X, torch.Tensor) and X.device != device:
         #   X = X.to(device)
        #if isinstance(y, torch.Tensor) and y.device != device:
         #   y = y.to(device)

        logits = model(X).prediction.squeeze(2)
        loss = self.basic_loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)

        return loss, preds

    def __call__(self, model, X, y):
        return self.calculate_loss(model, X, y)