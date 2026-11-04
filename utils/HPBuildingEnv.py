''' Simulation of heat-pump-office-building-interaction.'''

from data.heatpumps.heating_physics import PhysicalModel, FullState

class HPBuildingEnv():
    ''' Simulates the effect, a heat pump has on an office building.
    Each step is a transition from time point t to t+1.
    For building thermo dynamics, a physical model is utilized so that this env rather is a wraper.
    This env is not aiming to be a full RL environment, although it follows the Gymnasium structure.
    Adding a reward function, one can easily turn it into a RL env.
    '''
    def __init__(self, config, task_id):
        self.config = config
        # Task_id defines the parameters of the office building.
        self.params = task_id
        
        self.init_physical_model()

    def init_physical_model(self):
        ''' Initializes the physical model with all its components.'''
        # Todo: Initial core_state kommt aus HS-system.
        core_state = []
        full_state = FullState(core_state = core_state, params = self.params)
        self.physical_model = PhysicalModel(full_state = full_state,
                                            step_time_delta = 1,
                                            actions_per_step = 4)

    def step(self, action):
        ''' Calls the physical model and other components
        in order to make a time step.
        '''
        # Todo: Is this action call correct?
        self.physical_model.full_state.set_actions(action)
        self.physical_model.make_step()
        obs = self.physical_model.full_state.core_state 
        rew = 0 # as there is no reward function yet.
        done = False # None episodic env yet.

        return [obs, rew, done, {}]

    def reset(self):
        ''' Resets the core state, the action and thereby the physical model.'''
        self.physical_model.reset() # This should suffice.
        # Todo: Maybe, the reset input (core_state) should be varied. Now, it is a continuous time scale with dependencies.

        return self.physical_model.full_state.core_state 
