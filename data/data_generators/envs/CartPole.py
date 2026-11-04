''' Contains a CartPole wrapper anabling different environment physics.'''

from gym.envs.classic_control import CartPoleEnv


class VariableCartPole(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__()
        # Default parameters
        self.masscart = kwargs.get('masscart', 1.0)
        self.masspole = kwargs.get('masspole', 0.1)
        self.length = kwargs.get('length', 0.5)
        self.force_mag = kwargs.get('force_mag', 10.0)
        
        # Update physics
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def set_params(self, **params):
        """Update environment parameters without recreating"""
        # Update individual parameters
        if 'masscart' in params:
            self.masscart = params['masscart']
        if 'masspole' in params:
            self.masspole = params['masspole']
        if 'length' in params:
            self.length = params['length']
        if 'force_mag' in params:
            self.force_mag = params['force_mag']
            
        # Recalculate derived parameters
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
