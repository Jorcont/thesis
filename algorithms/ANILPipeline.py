''' Meta-Learning pipeline for the ANIL algorithm.'''

from typing_extensions import override

import torch.optim as optim
import learn2learn as l2l

from algorithms.GradientBasedPipeline import GradientBasedPipeline

class ANILPipeline(GradientBasedPipeline):
    """ This is the l2l meta-learning pipeline for the ANIL algorithm.
    It directly inherits from the general MAML-based pipeline and contains of only a few changes:
    - meta_model separated into a feature network and a head.
    - Only the head adapts during inner learning, while the feature network produces the data embeddings.
    - The meta_optimizer still updates the whole model, but with combination of feature and head params.
    - 
    """
    @override
    def init_algorithm(self):
        """ Initializes the meta-learning algorithm.
        This includes model wraping.
        """
        self.features = self.meta_model.features
        self.head = self.meta_model.classifier
        self.head = l2l.algorithms.MAML(self.head, lr=self.fast_lr)

    @override
    def configure_optimizers(self):
        """ Initializes all optimizers w.r.t. the underlying model components.
        In this example, one single optimizer suffices.
        But this will change for more complicated algorithms like PEARL."""
        all_parameters = list(self.features.parameters()) + list(self.head.parameters())
        self.meta_optimizer = optim.Adam(all_parameters, lr=self.meta_lr)

    @override
    def move_model_to_device(self):
        """ Moves both model parts to the device."""
        self.features.to(self.device)
        self.head.to(self.device)

    @override
    def learn_task_pool(self, pool, gradients=False):
        ''' Iterates over all tasks of the given pool and
        performs inner learning. All corresponding losses and/or metrics
        are saved in self.meta_losses.

        Args:
            - pool: The pool of tasks to be considered.
            gradients: Whether to propagate losses back to meta-optimizer.
                    True=meta-training; False=meta-val or test.
        '''
        self.meta_losses = []
        for task_id in pool:
            task = self.task_sampler.make_task(task_id=task_id, model=self.head.clone())
            # **ANIL**: Process input data with feature network before inner-learning.
            task.X_train = self.features(task.X_train)
            task.X_test = self.features(task.X_test)

            # Run task-specific learning and testing for particular task.
            self.task_losses = task.inner_learning()
            self.meta_losses.append(self.task_losses)
            if gradients:
                self.update_meta_loss()
