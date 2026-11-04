''' Basic Module for RL^2-type memory-based meta-learning.'''

import logging
from typing_extensions import override

from algorithms.LightningMAMLModule import LightningMAMLModule

logger = logging.getLogger("meta-netze")

class LightningRL2Module(LightningMAMLModule):
    ''' Simplest gradient-based class. 
    Theoretically speaking, this is no gradient-based learning, since inner learning does not exist.
    Rather than that, the "inner learning" is a simple gathering of context that changes the hidden state, attention etc. of the underlying memory-based model
    e.g. RNN, transformers etc.
    This idea is inspired by the RL^2 algorithm, which, however, is a meta-RL algo.
    Function changes simply set adapt to false for this purpose.
    '''
    @override
    def init_algorithm(self):
        """ Initializes the meta-learning algorithm.
        Only used for deactivating inner adaptation.
        """
        self.adapt = False
        self.higher = False # CAREFUL HERE. REMOVE ONLY WHEN FIXING THE WHOLE PART ABOUT HIGHER
        logger.info("Disabled model adaptation during inner learning.")

    @override
    def determine_inner_model(self):
        ''' Determines the inner model, what simply is the full model in RL^2.'''
        return self.meta_model
