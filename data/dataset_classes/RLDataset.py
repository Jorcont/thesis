''' Pytorch Dataset for RL tasks.'''

import random

from data.dataset_classes.SupervisedLearningDataset import SupervisedLearningDataset

class RLDataset(SupervisedLearningDataset):
    ''' This Dataset object is responsible for environment resets.
    It samples environment parameters from the parameter pool and uses them for reset.
    Its generator is hence an environment given by RLSampler class.
    '''
    def sample_env_params(self):
        ''' Samples an env configuration from the parameter pool.'''
        env_id = {}
        for key in self.pool.keys():
            env_id[key] = random.uniform(self.pool[key][0], self.pool[key][1])

        return env_id

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        ''' Randomly samples a specification from the parameter range.
        Then, the respective environment is reset and returned.
        Note that self.generator is a list of envs, in this version.
        '''
        spec = self.sample_env_params()
        self.envs[idx].set_params(spec)
        self.envs[idx].reset()
        return self.envs[idx]
