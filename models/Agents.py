''' Agents consisting of more than just a simple policy network.'''

from __future__ import annotations

import torch.nn as nn
from buffers.replay import ReplayBuffer, ContextBuffer # TODO: Woher soll das kommen?
from models.Encoders import ContextEncoder
# Todo: Not in utils yet.
from utils.common import soft_update, hard_update

from models.Critics import ContextQFunction
from models.Policys import GaussianContextPolicy

class PEARLSACAgent(nn.Module):
    """Pytorch lightning based Container that holds encoder, policy, critics.
    Note: The underlying critics, the policy and the context encoder are initialized in the
    model config and given by the config"""
    def __init__(
        self, config):
        super().__init__()
        self.context_encoder = config.context_encoder
        self.policy = config.policy
        # Critics for SAC.
        self.Q1 = config.Q1
        self.Q2 = config.Q2
        self.target_Q1 = config.target_Q1
        self.target_Q2 = config.target_Q2

        # Todo: What is this for? It probably belongs to the SAC algorithm.
        # hard_update(self.target_q1, self.q1)
        # hard_update(self.target_q2, self.q2)
