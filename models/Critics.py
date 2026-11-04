
from __future__ import annotations
import torch
import torch.nn as nn
from ..utils.common import weight_init # TODO: implement weight_init

class ContextQFunction(nn.Module):
    """Q(s,a,z) network."""
    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden_dims=(256,256)):
        super().__init__()
        in_dim = obs_dim + act_dim + z_dim
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, obs, act, z):
        x = torch.cat([obs, act, z], dim=-1)
        return self.net(x)
