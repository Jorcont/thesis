''' Different torch policy models.'''

from typing import Tuple
import torch
import torch.nn as nn

class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dims, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, act_dim)
        )

    def forward(self, x):
        # returns logits for Categorical distribution
        return self.net(x)

# Todo: What is this?
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianContextPolicy(nn.Module):
    """Squashed Gaussian policy pi(a|s,z)."""
    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden_dims=(256,256)):
        from utils.common import weight_init # TODO: implement function
        # Todo: This is not in utils yet.
        super().__init__()
        in_dim = obs_dim + z_dim
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, z], dim=-1)
        h = self.net(x)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs, z)
        std = log_std.exp()
        eps = torch.randn_like(std)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)
        # Log prob with tanh correction
        log_prob = (-0.5 * ((pre_tanh - mu) / (std + 1e-8)).pow(2) - log_std - 0.5 * torch.log(torch.tensor(2 * 3.141592653589793))).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs, z)
        return torch.tanh(mu)
