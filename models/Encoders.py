
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..utils.common import reparameterize, weight_init # TODO: Implement both functions


class TransitionEncoder(nn.Module):
    """Encode (s, a, r, s') into Gaussian stats for latent z_i ~ N(mu_i, sigma_i^2)."""
    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden_dims=(128,128)):
        super().__init__()
        in_dim = obs_dim + act_dim + 1 + obs_dim  # s, a, r, s'
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, z_dim)
        self.logvar = nn.Linear(last, z_dim)
        self.apply(weight_init)

    def forward(self, s, a, r, s2) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a, r, s2], dim=-1)
        h = self.net(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(min=-10.0, max=5.0)
        return mu, logvar

class ProductOfExperts(nn.Module):
    """Combine Gaussian experts (mu_i, logvar_i) into posterior via product of experts."""
    def __init__(self, z_dim: int, prior_logvar: float = 0.0):
        super().__init__()
        self.prior_logvar = prior_logvar
        self.z_dim = z_dim

    def forward(self, mus: torch.Tensor, logvars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mus/logvars: [B, K, z_dim]
        # Prior as an expert
        prior_mu = torch.zeros((mus.size(0), 1, self.z_dim), device=mus.device)
        prior_logvar = torch.full_like(prior_mu, self.prior_logvar)
        mus = torch.cat([mus, prior_mu], dim=1)
        logvars = torch.cat([logvars, prior_logvar], dim=1)
        var = logvars.exp()
        T = 1.0 / var  # precision
        mu_T = mus * T
        post_T = T.sum(dim=1)
        post_var = 1.0 / post_T
        post_mu = (mu_T.sum(dim=1)) * post_var
        post_logvar = post_var.log()
        return post_mu, post_logvar

class ContextEncoder(nn.Module):
    """Amortized inference network for PEARL producing posterior q(z|c)."""
    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden_dims=(128,128), prior_logvar: float = 0.1):
        super().__init__()
        self.tr_encoder = TransitionEncoder(obs_dim, act_dim, z_dim, hidden_dims)
        self.poe = ProductOfExperts(z_dim, prior_logvar)

    def infer_posterior(self, s, a, r, s2) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode each transition -> [K, z_dim]
        mu_i, logvar_i = self.tr_encoder(s, a, r, s2)
        # Add batch dimension if needed
        if mu_i.dim() == 2:
            mu_i = mu_i.unsqueeze(0)
            logvar_i = logvar_i.unsqueeze(0)
        mu, logvar = self.poe(mu_i, logvar_i)
        return mu, logvar

    def sample_z(self, s, a, r, s2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.infer_posterior(s, a, r, s2)
        z = reparameterize(mu, logvar)
        return z, mu, logvar
