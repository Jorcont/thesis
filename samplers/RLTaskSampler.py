''' General Task-Sampler to create and sample from task pool.'''

from typing import Any, Dict, List, Tuple
from typing_extensions import override

from torch.utils.data import DataLoader, Dataset

from samplers.SupervisedLearningTaskSampler import SupervisedLearningTaskSampler
from tasks.RLTask import RLTask

class RLTaskSampler(SupervisedLearningTaskSampler):
    ''' This is the TaskSampler for RL tasks.
    Among various similarities to the general TaskSampler, some certain details are different.
    In particular, various variables are named differently to better fit the RL paradigm and notions.

    Args:
        - config: Configures e.g. The numbers N, N_val and N_test of training, val and test tasks.
            Requires an additional env config.
    '''
    @override
    def init_pool_and_datasets(self):
        ''' Number of tasks (envs) must be assigned before creating pool of envs.
        '''
        self.set_number_of_tasks()
        self.create_pool_of_envs()
        self.split_pool()
        self.create_datasets()

    def create_pool_of_envs(self):
        """ Creates environment instances for meta-training, validation and testing
        so that they only need to be reset during meta-learning pipeline.
        """
        self.train_task_list = [self.config['env_config'].make_env() for _ in range(self.N)]
        self.val_task_list = [self.config['env_config'].make_env() for _ in range(self.N_val)]
        self.test_task_list = [self.config['env_config'].make_env() for _ in range(self.N_test)]

    @override
    def determine_split_idx(self, key):
        ''' Helper function for data spliting that returns split indices.
        This version splits intervals.
        '''
        upper_bound = self.task_pool[key][1]
        lower_bound = self.task_pool[key][0]
        interval_len = upper_bound - lower_bound
        # Calculate edge values for the 3er-split:
        lower_value = lower_bound + interval_len * self.splits[key][0]
        upper_value = lower_value + interval_len * self.splits[key][1]
        self.split_idx:List[int] = [ int(lower_value), int(upper_value)]

    @override
    def split_pool(self):
        """ Splits the task pool into training, validation and testing tasks,
        and creates the respective torch datasets."""
        for key in self.splits.keys():
            self.determine_split_idx(key)

            self.param_ranges['train'][key] = [self.task_pool[key][0], self.split_idx[0]]
            self.param_ranges['val'][key] = [self.split_idx[0], self.split_idx[1]]
            self.param_ranges['test'][key] = [self.split_idx[1], self.task_pool[key][1]]

    @override
    def create_datasets(self):
        """ Creates the pytorch datasets with the respective param_ranges.
        """
        super().create_datasets()
        # Set the generators as the respective list of envs.
        self.train_dataset.set_data_generator(self.train_task_list)
        self.val_dataset.set_data_generator(self.val_task_list)
        self.test_dataset.set_data_generator(self.test_task_list)

    @override
    def set_number_of_tasks(self):
        '''Assign number of tasks per meta-val/test.
        For this sampler, they must be given by the config.
        '''
        self.N_val = self.config.get('N_val', None)
        self.N_test = self.config.get('N_test', None)
        if (self.N_val is None) or (self.N_test is None):
            raise ValueError("You must define the number N_val of validation tasks and the number N_test of test tasks in the sampler config.")

    @override
    def get_in_and_out_dim(self) -> Tuple[int, int]:
        ''' For RL, in_dim means the dimensions of observation space,
        while out_dim is determined by the action space.

        Returns:
            obs_dim: The dimension of any flattened observation.
            act_dim: Dimension of actions.
        Note that the exact way how to yield dimensions might vary among envs.
        '''
        env = self.train_task_list[0]
        flattened_observation = env.reset()
        obs_dim = flattened_observation.shape[0]
        # Todo: action dimension is wrong this way.
        act_dim = env.action_space.dim

        return obs_dim, act_dim



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
        - config: Configuration of certain task-level hyperparams like adaption_steps. Also contains loss function.
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

        self.K = self.config['adaption_steps']
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
