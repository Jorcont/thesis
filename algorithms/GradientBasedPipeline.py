''' This module implements a general gradient-based meta-learning pipeline. '''

import torch.optim as optim
import learn2learn as l2l

class GradientBasedPipeline:
    """ This is the general gradient-based meta-learning pipeline
    orchestrating meta-training, -validation and -testing as well as task sampling and model updates.
    The l2l package is utilized for model updates.

    Note: This pipeline does not work for meta-RL.
    """
    def __init__(self, config):
        ''' Initializes the meta-learning pipeline.'''
        self.config = config
        self.meta_training_iter = config.meta_training_iter
        self.meta_lr = config.meta_lr
        self.fast_lr = config['task_config']['config'].inner_lr
        self.logging_level = config.logging_level

        self.val_every = config.validate_every
        self.print_meta_val = config.print_meta_val

        self.task_sampler = config.sampler

        self.device = config.device
        self.meta_model = self.config.head

        self.init_algorithm()
        self.move_model_to_device()
        self.configure_optimizers()

    def init_algorithm(self):
        """ Initializes the meta-learning algorithm.
        This includes model wraping.
        """
        # Todo: The algorithm should be given by config. This is particular MAML implementation.
        self.meta_model = l2l.algorithms.MAML(self.meta_model, lr=self.fast_lr, first_order=False)

    def configure_optimizers(self):
        """ Initializes all optimizers w.r.t. the underlying model components.
        In this example, one single optimizer suffices.
        But this will change for more complicated algorithms like PEARL."""
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=self.meta_lr)

    def move_model_to_device(self):
        """ Moves the model to the given device.
        This is likely to change from algorithm to algorithm."""
        self.meta_model = self.meta_model.to(self.device)

    def meta_training(self):
        """ Orchestrates meta-training by calling task-specific training."""
        for i in range(self.meta_training_iter):
            self.meta_optimizer.zero_grad() # Reset meta model gradients.
            train_task_list = self.task_sampler.train_dataset.sample_batch()

            # Call inner learning for all training tasks.
            # Note: Gradients=True means to propagate losses back to meta-optimizer.
            self.learn_task_pool(pool=train_task_list, gradients=True)

            # Use collected experience to update the model.
            self.meta_optimizer.step()

            # Meta-validation every l=val_every steps.
            if i % self.val_every == 0:
                self.meta_validation()

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
            task = self.task_sampler.make_task(task_id=task_id, model=self.meta_model.clone())
            # Run task-specific learning and testing for particular task.
            self.task_losses = task.inner_learning()
            self.meta_losses.append(self.task_losses)
            if gradients:
                self.update_meta_loss()

    def update_meta_loss(self):
        """ Handles the update of the meta-loss.
        This is determined by the meta-algorithm used and must hence be overwritten respectively.

        This particular example holds for MAML (and potentially other gradient-based meta-learners).
        The applied gradient averaging is not required but might stabilize training.
        Here, l2l is used as the underlying framework, so that
        backpropagation includes second-order derivatives of inner learning, as long as first_order=False.
        """
        scaled_loss = self.task_losses['test_loss']/ float(self.task_sampler.N)
        # Backpropagate scaled query loss into the meta-parameters:
        scaled_loss.backward()

    def meta_validation(self):
        """ Calls inner learning for all meta-validation tasks.
        Meta-Validation aims to evaluate the training progress throughout meta-training."""
        val_pool = self.task_sampler.val_dataset.sample_batch()
        self.learn_task_pool(pool=val_pool)
        if self.print_meta_val:
            self.calculate_performance_measures()

    def meta_testing(self):
        """ Calls inner learning for all meta-test tasks."""
        test_pool = self.task_sampler.test_dataset.sample_batch()
        self.learn_task_pool(pool=test_pool)
        self.calculate_performance_measures()

    def calculate_performance_measures(self):
        ''' Uses self.meta_losses to calculate performance measures over
        the task pool used for calculating these losses.
        '''
        pool_size = len(self.meta_losses)
        self.meta_gen_error = 0
        K = len(self.meta_losses[0]["train_losses"])
        self.meta_adaptation = [0 for _ in range(K)]
        for task_losses in self.meta_losses:
            train_losses = task_losses['train_losses']
            self.update_meta_adaptation(train_losses)
            test_loss = task_losses['test_loss']
            gen_loss = test_loss - train_losses[-1]
            self.meta_gen_error += gen_loss
        # Average all losses.
        self.meta_gen_error /= pool_size
        self.meta_adaptation = [loss/pool_size for loss in self.meta_adaptation]
        self.print_performance_measures()

    def update_meta_adaptation(self, train_losses):
        ''' Updates the list of train_losses used for adaptation speed calculation.
        '''
        i = 0

        for loss in train_losses:
            self.meta_adaptation[i] += loss
            i += 1

    def print_performance_measures(self):
        ''' Prints and plots all calculated performance measures in the desired form.
        '''
        # Todo: This should perhaps be a function that is configured.
        pass
