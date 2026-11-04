""" General Reinforcement Learning task parent class."""

from typing import final, Any, Dict, List, Tuple
import logging

import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

class RLTask:
    config: Dict[str, Any]
    ''' Represents a general episodic Reinforcement Learning (RL) task with respective connection
    to the corresponding environment and agent.
    One shot (K=1) is defined as one episode of agent-environment interaction.
    Parameters:
        - config: Configuration of certain task-level hyperparams like adaptation_steps. Also contains loss function.
        - task_id: Environment parameters defining the particular environment instance.
        - inner_learner: The task-level agent.
        device: The device (CPU or GPU) that is used for inner training.
    '''
    @final
    def __init__(self, task_id: Any, inner_learner: nn.Module):
        ''' Initializes this RL task.
        It is defined by the particular environment given as task_id
        (a choice that is taken to fit into the general meta-learning pipeline naming conventions).
        However, within this init, names are redefined to RL-specific notions for better clarity.
        For example, what is generally named task_id, is the actual env of this task.
        '''
        self.config = type(self).config
        self.env = task_id
        self.loss_fn = self.config['loss_fn']
        self.agent = inner_learner
        self.device = self.config['device']

        self.K = self.config['adaptation_steps']
        self.gamma = self.config['gamma']

    @final
    def inner_learning(self, adapt=True):
        ''' Orchestrates inner learning by calling self.K episodes of task-level learning.

        Returns: The total reward collected during task-level learning.
        '''
        returns = []
        losses = []
        for _ in range(self.K):
            self.run_single_episode(evaluation=not adapt)

            episode_reward = self.discount_rewards()
            returns.append(episode_reward)
            episode_loss = self.calculate_loss()
            losses.append(episode_loss)

            if adapt:
                self.adapt_agent()

        return {'train_returns': returns,
                'train_losses': losses,
                'test_loss': losses
                }

    def adapt_agent(self):
        ''' This function directly corresponds to the underlying model and RL algorithm.
        Most likely, it must be overwriten respectively for any use case.
        '''
        self.agent.adapt(self.episode_loss)

    def calculate_loss(self):
        ''' Calculates the current episode loss.
        This function highly depends on the inner RL algorithm and
        must, hence, be re-written for any special case,
        because the input of loss_fn depends on the loss choice.
        '''
        return self.loss_fn(self.episode_reward)
    
    @final
    def run_single_episode(self, evaluation=False):
        """ Runs one single episode with current model and collects sars-pairs.
        """
        obs = self.env.reset() # Initial state s_0
        self.episode_reward = []
        while not self.env.done:
            obs = self.preprocess_obs(obs)
            action_dist = self.agent(obs)
            action = self.post_process_action(action_dist, evaluation=evaluation)
            next_obs = self.env.step(action) # list of obs, rew, done, info.
            self.episode_reward.append(next_obs[1]) # add current reward.
            # Assign s_t+1 to current s_t i.e. transition of env.
            obs = next_obs[0]

    def pre_process_obs(self, obs):
        ''' Preprocessing of the current state.
        This must most likely be adapted to the particular pytorch policy model used.
        This example simply wraps the observation into a torch tensor.
        '''
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def post_process_action(self, action, evaluation=False):
        ''' Post-processes the action yielded from the agent.
        Most likely, must be adjusted to use case.
        This example assumes logits. It puts them into a torch categorical distribution.
        
        Args:
            evaluation: If true, action with highest probability mass is taken instead of random sampling.
        '''
        dist = torch.distributions.Categorical(logits=action)
        if evaluation:
            action = max()
        else:
            action = dist.sample()

        return action.item()

    @final
    def discount_rewards(self):
        """
        Compute discounted returns for a single episode.
        Returns a list of returns (same length as rewards), where each entry is the
        discounted sum from that timestep.
        """
        returns = []
        R = 0.0
        for r in reversed(self.episode_reward):
            R = r + self.gamma * R
            returns.insert(0, R) # Insert at the beginning of the list.
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize returns for stability (per-episode)
        if returns.std() != 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns
