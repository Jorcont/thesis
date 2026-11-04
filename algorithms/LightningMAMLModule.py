''' Pytorch Lightning Module for MAML.'''

from typing import final, Dict, List, Any
import numpy as np
from collections import defaultdict
import math
from typing import final, Dict, List, Any, Optional, Set
import logging

import higher
import learn2learn as l2l
import pytorch_lightning as pl
import torch
import time
import pandas as pd
from pathlib import Path
import os
from conf.configs import PipelineParams
from conf.sampler_configs import SamplerConfig
from conf.loss_configs import LossConfig
from algorithms.PipelineState import PipelineState as state
from utils.StepTimer import StepTimer
import torch.nn.functional as F
import logging
from evaluation.visualize_prototypes import PrototypeVisualization, plot_last_layer_weights

logger = logging.getLogger("meta-netze")

class LightningMAMLModule(pl.LightningModule):
    ''' This is the parent module for gradient-based meta-learning.
    It is adjusted to Model Agnostic Meta-Learning (MAML) as the most famous gradient-based meta-learner.
    This class receives all information from the main pipeline config created in configs.py
    (without receiving it as input, since PipelineParams is a global config class).

    Internal gradient-based meta-learning logic:
    Init (static): Defines attributes and calls init_algorithm and configure_optim functions.
    1. init_algorithm (changable): Wrapes the given model with meta-learning logic, e.g., with l2l wraper for MAML (as long as no higher is used).
    2. configure_optim (static): Configures optimizer dependent on the fast_lr and algorithm configurations. Divides into:
        a) Meta-Optimizer creation, where an ADAM optimizer is set for all meta-parameters.
        b) Inner optimizer, which is not needed for the default (l2l) case, but can be configured using hihger.
        If meta_lr or fast_lr are dicts instead of numbers, the higher package is activated in the init and configure_multi_lr_optim is called:
        configure_multi_lr_optim (static): Creation of learning rate groups for different model blocks. This results in different learning rates for different blocks and higher usage in inner learning.

    Train/val/test steps are defined for lightning. They use the global state-train attribute accordingly. All automatically call:
    learn_task_pool (static): General logic for learning a batch of tasks. Calls several customizable functions:
    1. unpack_and_encode_inputs (changable): Divides the current batch into X_train, X_test, y_train, y_test. Can be changed w.r.t. batch logic.
        But it is recommended to rather change the batches resulting from the use-case-specific dataset class.
    2. determine_inner_model (changable): Define the inner model, here. Very likely changes for different gradient-based algorithms.
    Dependent on meta-training or meta-testing stage:
    3. inner_learning (static): Performs classical inner learning with model adaptation, test/query performance measuring and metric tracking.
        a) Starts a higher inner loop for multi-lr case and ordinary inner_learning for normal one-lr cases. Hence, learning is itself an inner function, here.
        b) 0-shot performance in the testing case.
        c) Iteratively adapting to task-level training data and calculating test/query performance in meta-testing mode.
        d) Model adaptation in each loop based on the training loss:
        adapt_model: Adapts the model dependent on higher usage, l2l existence etc. This is no function which should be changed lightly.
        e) Final test/query performance calculation and tracking. Particularly important for meta-testing/evaluation of results.
    3'. extended_inner_learning (static): Wraper of inner_learning: Iteratively varies samples and adaptation steps and calls inner_learning each time. This is for evaluating meta-performance among tasks.
        Requires four functions from the use-case-specific dataset class (and is hence not ment to be changed):
        a) copy_batch (changable): How to copy any X or y. Varies due to data shapes/types.
        b) get_emptyX/get_emptyY (changable): Returns an empty X or y object.
        c) add_one_sample (changable): Manges how to add one single sample to X and the corresponding "label" to y.
        Together, these function enable extended_inner_learning to examine performance for all K shots individually and for each adaptation step.
    '''

    @final
    def __init__(self):
        ''' Initializes the meta-learning pipeline.'''
        super().__init__()
        self.config = PipelineParams.get_config()
        self.sampler = self.config.sampler
        self.meta_lr = self.config.meta_lr
        self.fast_lr = self.config.fast_lr # Todo: Move inner_lr from task_config to fast_lr in PipelineParams
        self.higher = False
        self.meta_lrs = None
        self.weight_decay = self.config.weight_decay
        self.adaptation_steps: int = self.config.adaptation_steps
        self.print_meta_val: bool = self.config.print_meta_val
        self.meta_model = self.config.model
        self.data_parallel = self.config.data_parallel
        self.results_cache: List[List[Any]] = []
        self.pred_cache = []  # stores (preds_tensor, y_test_tensor) per test task
        self.task_sampler = SamplerConfig.get_config().sampler
        self.loss_fn = LossConfig.get_config().loss_fn
        self.adapt = True # Normal case where there is adaptation within every task.
        self.inner_optimizer = None
        self.automatic_optimization = False
        self.first_order = self.config.first_order
        self.metrics = self.config.metrics
        self.inner_batch_size = self.config.inner_batch_size
        self.interpretation_cache = []
        self.plant_id_cache = []
        self._masked_cache = []
        self.time_cache = []
        self.support_cache = []
        self.support_time_cache = []
        self._suppress_pred_cache = False
        self.zero_target_threshold = self.config.zero_target_threshold
        self._per_plant_test_cache = []

        self.grad_log_every_n_steps = self.config.grad_log_every_n_steps
        self.grad_hist_every_n_steps = self.config.grad_hist_every_n_steps
        self.log_grad_histograms = self.config.log_grad_histograms

        self._loop_timer = StepTimer(["unpack", "clone", "inner"])
        self._step_timer = StepTimer(["loss", "adapt"])
        if isinstance(self.fast_lr, Dict) or isinstance(self.meta_lr, dict):
            self.higher = True
            logger.info("Enabled higher for configurable learning rates")

        self.init_algorithm()
        self.configure_optimizers()
        self.current_task = -1

    def init_algorithm(self):
        """ Initializes the meta-learning algorithm.
        This includes model wraping.
        """
        base_lr = self.fast_lr if isinstance(self.fast_lr, (int, float)) else 1e-3
        self.meta_model = l2l.algorithms.MAML(
            self.meta_model, # all parameters
            lr=base_lr, # inner learning rate
            first_order=self.first_order # FOMAML - no Hessian needed
            )
        # Better option, but we use different lr with "higher"
        # self.meta_model = l2l.algorithms.MAML(self.meta_model, lr=self.fast_lr, first_order=False)

    def unpack_model(self) -> torch.nn.Module:
        """Returns the underlying model, unwrapping the l2l MAML wrapper if present.
        Requires the model to implement `init_last_layer`.
        """
        net = self.meta_model
        if hasattr(net, "module"):  # l2l.algorithms.MAML stores the wrapped net here
            net = net.module

        return net

    def configure_optimizers(self):
        ''' Configures optimizers for meta-training.
        The function name is required by pytorch_lightning.'''

        if self.config.always_init_last_layer:
            model = self.unpack_model()
            # Parameter ids of the last layer -> must be excluded from the meta-optimizer.
            last_layer_param_ids = {id(p) for p in model.last_layer.parameters()}
        else:
            last_layer_param_ids = set()

        if isinstance(self.meta_lr, dict):
            logger.debug("More than     one learning rate for meta-optimizer. Creating multi-lr meta-optimizer.")
            self.meta_optimizer = self.configure_multi_lr_optim(
                lr_dict=self.meta_lr, exclude_param_ids=last_layer_param_ids
            )
        else:
            logger.debug("Creating ordinary meta-optimizer (ADAM) without the last layer.")
            meta_params = [
                p for p in self.meta_model.parameters() if id(p) not in last_layer_param_ids
            ]
            self.meta_optimizer = torch.optim.Adam(
                meta_params, lr=self.meta_lr, weight_decay=self.weight_decay
            )

        if isinstance(self.fast_lr, dict):
            logger.debug("More than one learning rate for inner optimizer. Creating multi-lr inner optimizer.")
            self.inner_optimizer = self.configure_multi_lr_optim(lr_dict=self.fast_lr)
            #self.meta_lrs = MetaLearningRates(self.fast_lr, device=self.device)
            ## Meta-Optimizer must also update these parameters.
            #self.meta_optimizer.add_param_group(
            #{"params": self.meta_lrs.parameters(), "lr": self.meta_lr if not isinstance(self.meta_lr, dict) else list(self.meta_lr.values())[0]})
            #self.inner_optimizer = None

        return self.meta_optimizer

    def configure_multi_lr_optim(self, lr_dict: dict, exclude_param_ids: Optional[set] = None):
        ''' Configures an optimizer with more than one single learning rate.
        Optionally excludes given parameter ids (e.g. the last layer for the meta-optimizer).
        '''
        exclude_param_ids = exclude_param_ids or set()
        model = self.unpack_model()

        group = []
        for lr_name, lr_value in lr_dict.items():
            param_name = lr_name
            attr = getattr(self.meta_model, param_name)
            if isinstance(attr, torch.nn.Module):
                param = attr.parameters()
            elif isinstance(attr, torch.nn.Parameter):
                param = [attr]
            else:
                raise TypeError(f"'{param_name}' is neither a Module nor a Parameter")
            group.append({
                "params": param,
                "lr": lr_value
            })

        logger.debug("The respective param group for multi-lr optimizer is {group}")
        return torch.optim.Adam(group, weight_decay=self.weight_decay)

    def update_optimizer_last_layer(self, optimizer: torch.optim.Optimizer, group_index: Optional[int] = None) -> None:
        """
        Replaces the last_layer parameters inside the optimizer after init_last_layer()
        created a brand-new layer. Other parameter groups and their optimizer state
        are left untouched.
        Note: This function is ment to work with multible learning rates.
            It is supposed to remove the last layer from the meta-optimizer.
            The inner optimizer is changed for each task accordingly.

        Args:
            optimizer: the optimizer holding the last_layer parameters.
            group_index: index of the param group that contains the last_layer params.
                        If None, the group is detected automatically.
        """
        new_params = list(self.last_layer.parameters())
        new_param_ids = {id(p) for p in new_params}

        # 1) Find the group holding the last_layer params (auto-detect if needed)
        if group_index is None:
            for i, group in enumerate(optimizer.param_groups):
                # A group belongs to last_layer if none of its params are still
                # referenced by the model outside last_layer.
                other_ids = {id(p) for p in self.parameters()} - new_param_ids
                if all(id(p) not in other_ids for p in group["params"]):
                    group_index = i
                    break
            if group_index is None:
                raise ValueError("Could not auto-detect the last_layer param group.")

        group = optimizer.param_groups[group_index]

        # 2) Drop stale optimizer state for the old params
        for old_p in group["params"]:
            optimizer.state.pop(old_p, None)

        # 3) Swap in the new params (fresh state will be created on next step)
        group["params"] = new_params

    def training_step(self, batch, batch_idx):
        ''' Required training function for pytorch lightning.'''
        # torch.cuda.empty_cache()
        state.train = True
        logger.debug(f"Current training status: {state.train}")

        opt = self.optimizers()
        opt.zero_grad()

        meta_loss = self.learn_task_pool(pool=batch)
        
        # NEW: batch-level train diagnostics from the inner loop
        zero_shot_avg = self._zero_shot_loss_sum / max(self._zero_shot_loss_count, 1)
        zero_shot_masked_avg = self._zero_shot_loss_masked_sum / max(self._zero_shot_loss_masked_count, 1)
        meta_loss_masked_avg = self._meta_loss_masked_sum / max(self._meta_loss_masked_count, 1)
        final_support_avg = self._final_support_loss_sum / max(self._final_support_loss_count, 1)
        final_support_masked_avg = self._final_support_loss_masked_sum / max(self._final_support_loss_masked_count, 1)

        # ── Gradient clipping ─────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.trainer.gradient_clip_val or 1.0,
        )
        opt.step() # θ ← θ - β·∇
        self.log(
            "effective_batch_size",
            float(self.trainer.accumulate_grad_batches * len(batch)),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "meta_train_loss",
            meta_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "meta_train_zero_shot_loss",
            zero_shot_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        if self._zero_shot_loss_masked_count > 0:
            self.log(
                "meta_train_zero_shot_loss_masked",
                zero_shot_masked_avg,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=self.data_parallel,
            )

        if self._meta_loss_masked_count > 0:
            self.log(
                "meta_train_loss_masked",
                meta_loss_masked_avg,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=self.data_parallel,
            )

        if self._final_support_loss_count > 0:
            self.log(
                "meta_train_support_loss_final",
                final_support_avg,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=self.data_parallel,
            )

        if self._final_support_loss_masked_count > 0:
            self.log(
                "meta_train_support_loss_final_masked",
                final_support_masked_avg,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=self.data_parallel,
            )

        torch.cuda.empty_cache()

        # Diagnostic for monitoring the loss and gradients
        if batch_idx % self.hparams.get("grad_log_every_n_steps", 50) == 0:# and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Epoch {self.current_epoch} Step {batch_idx} | meta_loss={meta_loss:.6f}")
            for name, p in self.meta_model.named_parameters():
                if p.grad is not None:
                    logger.debug(f"  grad {name}: {p.grad.norm():.6f}")
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"⚠️  NaN/Inf detected in: {name}")

            # Weight norms per layer
            for name, p in self.meta_model.named_parameters():
                print(f"  weight {name:40s}: {p.data.norm():.6f}")

            # Check if parameters are actually moving
            for name, p in self.meta_model.named_parameters():
                if p.grad is not None:
                    print(f"{name}: grad_norm={p.grad.norm():.8f}")
        return meta_loss


    def validation_step(self, batch, batch_idx):
        """Required validation function for pytorch_lightning."""
        state.train = not self.print_meta_val # Needs to be FALSE
        # state.train = False 
        logger.debug(f"Current training status: {state.train}")

        with torch.enable_grad():
            meta_loss = self.learn_task_pool(pool=batch)

        if meta_loss.item() == 0.0:
            print(f"⚠️ [META-VAL] batch {batch_idx}: meta_loss=0! pool_size={len(batch)}")
            X_train, y_train, X_test, y_test = batch[0]
            if isinstance(X_test, dict):
                print(f"    X_test keys: {list(X_test.keys())}, encoder_lengths shape: {X_test['encoder_lengths'].shape}")
            print(f"    y_test shape: {y_test.shape}, y_test sum: {y_test.sum()}")
        
        meta_loss_masked_avg = self._meta_loss_masked_sum / max(self._meta_loss_masked_count, 1)
        zero_shot_masked_avg = self._zero_shot_loss_masked_sum / max(self._zero_shot_loss_masked_count, 1)
        zero_shot_avg = self._zero_shot_loss_sum / max(self._zero_shot_loss_count, 1)

        ''' Required validation function for pytorch_lightning.'''

        self.log(
            "meta_val_loss",
            meta_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "meta_val_loss_masked",
            meta_loss_masked_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "zero_shot_val_loss",
            zero_shot_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "zero_shot_val_loss_masked",
            zero_shot_masked_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )
        return meta_loss

    @final
    def test_step(self, batch, batch_idx):
        ''' Test step function in the form used by pytorch lightning.
        '''
        # Note: Even for trainer.inference_mode = False, this line is required for testing.
        torch.set_grad_enabled(True)
        state.train = False
        logger.debug(f"Current training status: {state.train}")
        meta_loss = self.learn_task_pool(pool=batch)

        # zero_shot_avg = self._zero_shot_loss_sum / max(len(batch), 1)
        zero_shot_avg = self._zero_shot_loss_sum / max(self._zero_shot_loss_count, 1)

        meta_loss_masked_avg = self._meta_loss_masked_sum / max(self._meta_loss_masked_count, 1)
        zero_shot_masked_avg = self._zero_shot_loss_masked_sum / max(self._zero_shot_loss_masked_count, 1)

        self.log(
            "meta_test_loss",
            meta_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "meta_test_loss_masked",
            meta_loss_masked_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "zero_shot_test_loss",
            zero_shot_avg,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )

        self.log(
            "zero_shot_test_loss_masked",
            zero_shot_masked_avg,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=self.data_parallel,
        )
        return meta_loss  

    @final
    def learn_task_pool(self, pool):
        ''' Iterates over all tasks of the given pool and
        performs inner learning. All corresponding losses and/or metrics
        are saved in self.meta_losses.

        Args:
            - pool: The pool of tasks to be considered.
        '''
        self.meta_loss = 0
        self._zero_shot_loss_sum = 0.0
        self._zero_shot_loss_count = 0
        self._zero_shot_loss_masked_sum = 0.0
        self._zero_shot_loss_masked_count = 0
        self._meta_loss_masked_sum = 0.0
        self._meta_loss_masked_count = 0
        # final support/train loss after adaptation
        self._final_support_loss_sum = 0.0
        self._final_support_loss_count = 0
        self._final_support_loss_masked_sum = 0.0
        self._final_support_loss_masked_count = 0
        self._pool_size = len(pool)
        self._loop_timer.reset()
        self._step_timer.reset()

        t_pool = time.perf_counter()

        for i, batch in enumerate(pool):
            self.current_task = i
            logger.debug(f"Current task: {i}")
            # self.X_train, self.y_train, self.X_test, self.y_test = self.unpack_and_encode_inputs(batch)
            if self.config.always_init_last_layer:
                # Number of classes for the current task (derived from the support labels).
                num_classes = int(torch.unique(self.y_train).numel())
                self.reinit_last_layer(num_classes=num_classes)
            # inner_model = self.determine_inner_model()
            # if not state.train:
            #     self.results_cache.append([])
            #     self.extended_inner_learning()
            #     continue
            # self.inner_learning(inner_model)

            with self._loop_timer("unpack"):
                self.X_train, self.y_train, self.X_test, self.y_test = \
                    self.unpack_and_encode_inputs(batch)

            with self._loop_timer("inner"):
                if not state.train:
                    self.results_cache.append([])
                    self.extended_inner_learning()   # creates its own inner_model internally
                else:
                    with self._loop_timer("clone"):
                        inner_model = self.determine_inner_model()
                    self.inner_learning(inner_model)
                    del inner_model                  # ← only exists on train path

            # ── Always reached for both train and val/test ────────────────
            del self.X_train, self.y_train, self.X_test, self.y_test
        
        arr = np.asarray(self._masked_cache, dtype=float)
        n_valid = int(np.count_nonzero(~np.isnan(arr)))
        masked_mean = float(np.nanmean(arr)) if n_valid > 0 else float('nan')
        logger.debug(f"  [pool masked] mean={masked_mean:.6f} over {n_valid}/{len(arr)} tasks")
        pool_ms = (time.perf_counter() - t_pool) * 1e3
        if self._loop_timer.has_data("inner") and logger.isEnabledFor(logging.DEBUG):
            print(
                f"\n[LightningMAMLModule] learn_task_pool ({len(pool)} tasks):\n"
                f"  total pool time : {pool_ms:.1f} ms\n"
                f"  unpack  avg/max : {self._loop_timer.avg_max_report('unpack')}\n"
                f"  clone   avg/max : {self._loop_timer.avg_max_report('clone')}\n"
                f"  inner   avg/max : {self._loop_timer.avg_max_report('inner')}\n"
                f"  forward+loss    : {self._step_timer.total_pct_report('loss',  pool_ms)}\n"
                f"  adapt (backward): {self._step_timer.total_pct_report('adapt', pool_ms)}"
            )
        result = self.meta_loss/len(pool)
        return torch.tensor(result, requires_grad = False)

    def reinit_last_layer(self, num_classes):
        ''' Re-initializes the PPNet last layer for the current task's class count
        and rebuilds the inner optimizer so `higher` binds to the fresh parameters.
        '''
        model = self.unpack_model()
        # Brand-new last layer (no weight carry-over).
        model.init_last_layer(num_classes=num_classes)
        model.to(self.device)

        # Rebuild the inner optimizer over the (new) parameters.
        if isinstance(self.fast_lr, dict):
            self.inner_optimizer = self.configure_multi_lr_optim(lr_dict=self.fast_lr)

    def determine_inner_model(self):
        ''' Determines the inner model, what simply is the full model in MAML.'''
        if self.higher:
            return self.meta_model
        else:
            inner = self.meta_model.clone()
            inner.train()  # ← cuDNN RNN backward requires training mode
            return inner

    @final
    def inner_learning(self, inner_model) -> None:
        """Task-level adaptation and query evaluation."""
        
        def learning(model, optimizer=None):
            """
            Runs zero-shot evaluation, the inner adaptation loop (support data),
            and the final support/query loss computation for a single task."""
            if not state.train:
                self.metrics.init_metrics()

            # ── Zero-shot: ALWAYS compute, not just in DEBUG ──────────────
            with torch.no_grad():
                zero_shot_error, zero_shot_preds = self.loss_fn(model, self.X_test, self.y_test)
            zero_shot_val = zero_shot_error.item()

            # After computing zero_shot_val, add:
            zero_shot_masked_val = self._masked_error(
                zero_shot_preds, self.y_test, self.zero_target_threshold
            )
            self.metrics.task_metrics['zero_shot_loss_masked'] = zero_shot_masked_val  # need to keep preds from zero-shot

            self._zero_shot_loss_sum += zero_shot_val   # accumulate across tasks
            self._zero_shot_loss_count += 1

            if not np.isnan(zero_shot_masked_val):
                self._zero_shot_loss_masked_sum += zero_shot_masked_val
                self._zero_shot_loss_masked_count += 1

            if not state.train:
                # self.metrics.task_metrics['zero_shot_loss'] = zero_shot_val
                # self.metrics.task_metrics['zero_shot_loss_masked'] = self._masked_error(
                #     zero_shot_preds, self.y_test, self.zero_target_threshold)
                self.metrics.task_metrics['zero_shot_loss_masked'] = zero_shot_masked_val

            # else:
            #     self._zero_shot_loss_sum += zero_shot_val
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                        f"  [task {self.current_task} | zero-shot query] "
                        f"test_loss={zero_shot_val:.6f}")

            # ── Adaptation steps ──────────────────────────────────
            if not self._support_is_empty():
                for step in range(self.adaptation_steps):
                    if not state.train:
                        with torch.no_grad():
                            test_error, test_preds = self.loss_fn(
                                model, self.X_test, self.y_test)

                        # ── max error over 96 query time steps ──────────────
                        with torch.no_grad():
                            p = test_preds[0] if isinstance(test_preds, (tuple, list)) else test_preds
                            p = p.detach()
                            yt = self.y_test.detach()
                            # TFT quantile output: pick median channel (index 1)
                            if p.dim() == yt.dim() and p.shape[-1] != yt.shape[-1]:
                                p = p[..., 1:2]                        # [B, T, 1]
                            per_step_errors = (p - yt).abs()           # MAE per step, shape [B, T] or [B, T, 1]
                            query_loss_max = per_step_errors.max().item()
                        self.metrics.task_metrics['query_loss_max'] = query_loss_max
                        # ──────────────────────────────────────────────────────────────

                        self.metrics.compute_metrics(
                            test_preds, self.y_test, test_error.item(), mode='test')

                    with torch.enable_grad():
                        train_error, train_preds = self.loss_fn(
                            model, self.X_train, self.y_train)

                    # ── Debug: inner step diagnostics ─────────────────
                    if logger.isEnabledFor(logging.DEBUG):
                        preds = train_preds[0] if isinstance(train_preds, tuple) else train_preds
                        logger.debug(
                            f"  [task {self.current_task} | step {step}] "
                            f"train_loss={train_error.item():.6f} | "
                            f"pred=[{preds.min():.4f}, {preds.max():.4f}] | "
                            f"target=[{self.y_train.min():.4f}, {self.y_train.max():.4f}]"
                        )
                    # ──────────────────────────────────────────────────

                    if not state.train:
                        self.metrics.compute_metrics(
                            train_preds, self.y_train, train_error.item())

                    if self.adapt:
                        model = self.adapt_model(model, train_error, optimizer)

                    del train_error, train_preds

            # ── Final query evaluation ────────────────────────────
            if not state.train:
                with torch.no_grad():
                    test_error, test_preds = self.loss_fn(
                        model, self.X_test, self.y_test)
                self.metrics.compute_metrics(
                    test_preds, self.y_test, test_error.item(), mode='test')
                
                # ── NEW: parallel masked metric (report-only) ──────────────
                # self.metrics.task_metrics['test_loss_masked'] = self._masked_error(
                #     test_preds, self.y_test, self.zero_target_threshold)
                
                # self._masked_cache.append(self._masked_error(
                #     test_preds, self.y_test, self.zero_target_threshold))

                test_loss_masked_val = self._masked_error(
                test_preds, self.y_test, self.zero_target_threshold
                )
                self.metrics.task_metrics['test_loss_masked'] = test_loss_masked_val
                self._masked_cache.append(test_loss_masked_val)

                if not np.isnan(test_loss_masked_val):
                    self._meta_loss_masked_sum += test_loss_masked_val
                    self._meta_loss_masked_count += 1

                if logger.isEnabledFor(logging.DEBUG):
                    yq = self.y_test.detach()
                    logger.debug(
                        f"  [task {self.current_task} | query] "
                        f"test_loss={test_error.item():.6f} | "
                        f"masked={self.metrics.task_metrics['test_loss_masked']:.6f} | "
                        # f"max_loss={query_loss_max:.6f} | "    # ← add this
                        f"max_loss={locals().get('query_loss_max', float('nan')):.6f} |"
                        f"query_target=[{yq.min():.4f}, {yq.max():.4f}] | "
                        f"n>{self.zero_target_threshold}={int((yq > self.zero_target_threshold).sum())}"
                    )
                # ───────────────────────────────────────────────────────────

                self.metrics.task_metrics["k_shot"] = getattr(self, "_current_k", None)
                # self.results_cache[self.current_task].append(self.metrics.task_metrics)
                self.results_cache[-1].append(self.metrics.task_metrics)

                # ── Per-plant cache (only by real full-shot, not by k-shot loops) ──

                if (getattr(self, "_per_plant_test_cache", None) is not None
                    and not getattr(self, "_suppress_pred_cache", True)
                    and not self._support_is_empty()):
                    dataset = self.task_sampler.test_dataset
                    self._per_plant_test_cache.append({
                        "plant_id":       getattr(dataset, "_last_plant_id", f"task_{len(self._per_plant_test_cache)}"),
                        "rated_power_kW": getattr(dataset, "_last_rated_power_kW", None),
                        "zero_shot_MAE":  self.metrics.task_metrics.get("zero_shot_loss_masked"),
                        "adapted_MAE":    self.metrics.task_metrics.get("test_loss_masked"),
                    })


                dataset = self.task_sampler.test_dataset
                if not getattr(self, "_suppress_pred_cache", True):
                    self.pred_cache.append(
                        (test_preds.detach().cpu(), self.y_test.detach().cpu()))
                    self.plant_id_cache.append({
                        "plant_id":       getattr(dataset, '_last_plant_id', None),
                        "rated_power_kW": getattr(dataset, '_last_rated_power_kW', None),
                    })
                    self.time_cache.append( getattr(dataset, '_last_timestamps', None))
                    
                    # query timestamps — shape [n_win, look_ahead] or [n_win * look_ahead]
                    # self.time_cache.append(
                    #     getattr(dataset, 'query_timestamps', None)   # ← replace with correct attr name
                    # )

                    # support timestamps — same idea
                    self.support_time_cache.append(
                        getattr(dataset, 'support_timestamps', None)  # ← replace with correct attr name
                    )
                    self.support_cache.append((
                        None, 
                        self.y_train.detach().cpu()))
                self._capture_interpretation()
                self.meta_loss += test_error.detach().item()
            else:
                # NEW: final support/train loss AFTER adaptation, report-only
                if not self._support_is_empty():
                    with torch.no_grad():
                        final_train_error, final_train_preds = self.loss_fn(
                            model, self.X_train, self.y_train
                        )

                    final_train_val = final_train_error.item()
                    self._final_support_loss_sum += final_train_val
                    self._final_support_loss_count += 1

                    final_train_masked_val = self._masked_error(
                        final_train_preds, self.y_train, self.zero_target_threshold
                    )
                    if not np.isnan(final_train_masked_val):
                        self._final_support_loss_masked_sum += final_train_masked_val
                        self._final_support_loss_masked_count += 1

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"  [task {self.current_task} | final support] "
                            f"train_loss={final_train_val:.6f} | "
                            f"masked={final_train_masked_val:.6f}"
                        )

                    del final_train_error, final_train_preds
                test_error, test_preds = self.loss_fn(model, self.X_test, self.y_test)
                (test_error / self._pool_size).backward()   # ← per-task backward
                self.meta_loss += test_error.detach().item()

                # NEW: masked final query loss for reporting only
                test_loss_masked_val = self._masked_error(
                    test_preds.detach(), self.y_test, self.zero_target_threshold
                )
                if not np.isnan(test_loss_masked_val):
                    self._meta_loss_masked_sum += test_loss_masked_val
                    self._meta_loss_masked_count += 1

                # ── Debug: query loss after adaptation ────────────
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"  [task {self.current_task} | query] "
                        f"test_loss={test_error.item():.6f} | "
                        f"masked={test_loss_masked_val:.6f}"
                    )
                # ──────────────────────────────────────────────────
            del test_error

        if self.higher:
            # higher tracks second-order gradients through the inner loop
            with higher.innerloop_ctx(
                inner_model,
                opt=self.inner_optimizer,
                track_higher_grads=False, # FOMAML
                copy_initial_weights=False,
            ) as (higher_model, higher_optimizer):
                learning(higher_model, higher_optimizer)
        else:
            learning(inner_model)

    def unpack_and_encode_inputs(self, batch):
        ''' Performs a forward pass through backbone if it exist.
        Just a placeholder in this version.'''
        X_train, y_train, X_test, y_test = batch
        return X_train, y_train, X_test, y_test
    
    def adapt_model(self, inner_model, loss, inner_optimizer=None):
        """ Adapts the model.
        It depends on the meta-learning algorithm/library used.
        This version corresponds to l2l meta models.
        But higher usage is also enabled if more than just one inner learning rate exists.
        """
        if self.meta_lrs is not None:
            inner_model = self._adapt_per_group(loss, inner_model)
        elif self.higher:
            inner_optimizer.step(loss)
        else: # Means using l2l's adapt function (simple case without higher).
            inner_model.module.training = True
            inner_model.adapt(loss, allow_nograd=True, allow_unused=True)
        return inner_model
    
    def compute_grads_with_higher(self, loss, inner_model):
        ''' Computes gradients and uses them to update each individual part of the model.
        Iterates over model blocks and assign corresponding parameters to it.
        Only works when higher is True, i.e. when gradient computation for inner learning is enabled.
        '''
        if not self.higher:
            raise ValueError("Wrong function call! Higher must be used to use this function")
        grads = torch.autograd.grad(
            loss,
            inner_model.parameters(),
            create_graph=True,  # important for meta-gradients
                        allow_unused=True # Necessary if backbone has unused layers
        )
        new_params = {}
        for (name, param), grad in zip(inner_model.named_parameters(), grads):
            if grad is None: # since allow_unused = True grad can be none
                new_params[name] = param
                continue
            group_name = self.get_group_name_for_param(name)
            if self.meta_lrs is not None and group_name in self.meta_lrs.lrs:
                lr = self.meta_lrs.get_lr(group_name)
            else: # Fallback: A fixed value.
                lr = torch.tensor(self.fast_lr if isinstance(self.fast_lr, (float, int)) else 1e-2, device=param.device)
            new_params[name] = param - lr * grad

        inner_model.update_params(new_params)

        return inner_model
    
    def extended_inner_learning(self):
        ''' Wraps inner learning with another loop which iterates over samples
        in order to also determine sample efficiency.
        This function can be modified if the user wants even more information during testing.'''
        # ── Save full X and y ─────────────────────────────────────────
        if isinstance(self.X_train, dict):
            # pytorch_forecasting TFT dict mode
            all_samples = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in self.X_train.items()}
            n_samples = all_samples["encoder_lengths"].size(0)
        elif isinstance(self.X_train, (tuple, list)) and not isinstance(self.X_train, torch.Tensor):
            # TFT mode: (X_enc, X_dec) tuple
            all_samples = tuple(x.clone() for x in self.X_train)
            n_samples = all_samples[0].size(0)
        else:
            # LSTM mode: plain tensor
            all_samples = self.X_train.clone()
            n_samples = all_samples.size(0)
        # print(f"[extended_inner_learning] n_samples={n_samples}")
        do_sample_eff = self.metrics.calc_sample_efficiency and self._is_testing()

        all_y = self.y_train.clone()  # ← save full y
        _accumulated = self.meta_loss  # save accumulation from previous tasks

        # if not self.metrics.calc_sample_efficiency:
        if not do_sample_eff:
            print("Starting if statement in extended_inner_learning")
            #  ── Full-shot evaluation: adapt on ALL support data at once ────
            self.inner_model = self.determine_inner_model()
            self.X_train = all_samples
            self.y_train = all_y
            self.meta_loss = 0
            self.inner_learning(self.inner_model)
            # ── Restore accumulation + add this task's contribution ──
            self.meta_loss = _accumulated + self.meta_loss
        else:
            # print(f"[extended_inner_learning] n_samples={n_samples}")

            val_loss_sum = 0.0   # separate counter
            val_loss_count = 0
            original_steps = self.adaptation_steps
            # ── Zero-shot: no adaptation data ─────────────────────────────
            self._current_k = None
            self.inner_model = self.determine_inner_model()
            self.X_train = self._slice_samples(all_samples, 0)
            self.y_train = all_y[:0]
            self.adaptation_steps = 0  # no data → no adaptation
            self.meta_loss = 0
            self.inner_learning(self.inner_model)
            
            # ── Restore accumulation + add this task's contribution ──
            self._suppress_pred_cache = True

            k_values = self._get_eval_k_values(n_samples)
            for k in k_values:
                self.inner_model = self.determine_inner_model()
                self.X_train = self._slice_samples(all_samples, k)
                self.y_train = all_y[:k]
                # Scale steps: prevent overfitting at small k
                self.adaptation_steps = original_steps # min(original_steps, max(1, k // 3))
                self.meta_loss = 0
                self._current_k = k
                self.inner_learning(self.inner_model)

            # Restore
            self._suppress_pred_cache = False
            self.adaptation_steps = original_steps
            self.inner_model = self.determine_inner_model()
            self.X_train = self._slice_samples(all_samples, n_samples)
            self.y_train = all_y
            self.meta_loss = 0
            self._current_k = n_samples
            self.inner_learning(self.inner_model)
            self.meta_loss = _accumulated + self.meta_loss

        del all_samples, all_y

    def on_validation_epoch_start(self):
        """Clear caches at the start of each validation epoch to prevent unbounded growth."""
        self._suppress_pred_cache = True
        torch.cuda.empty_cache()
        self.pred_cache.clear()
        self.results_cache.clear()
        self.interpretation_cache.clear()  
        self.plant_id_cache.clear()
        self.support_cache.clear()
        self._masked_cache = []
        self.time_cache.clear()
        self.support_time_cache.clear()

    def on_test_epoch_start(self):
        """Clear caches at the start of testing to ensure clean state."""
        self._suppress_pred_cache = False
        self.pred_cache.clear()
        self.results_cache.clear()
        self.interpretation_cache.clear()
        self.plant_id_cache.clear()
        self._masked_cache = []
        self.support_cache.clear()
        self.time_cache.clear()
        self.support_time_cache.clear()
        self._per_plant_test_cache = []

    def on_validation_epoch_end(self):
        """Lightning hook: prints the epoch-averaged meta-validation loss to stdout."""
        avg_loss = self.trainer.callback_metrics.get("meta_val_loss")
        if avg_loss is not None:
            print(
                f"[META-VAL] Epoch {self.current_epoch} | "
                f"avg meta_loss = {avg_loss.item():.6f}"
            )

    def on_test_epoch_end(self):
        """
        Lightning hook: aggregates meta-test results (zero-shot vs. post-adaptation
        loss, per-plant adaptation gain) and triggers KS-distribution analysis and
        prediction-plot export at the end of meta-testing.
        """
        # ── Temporary: inspect plant_id_cache structure ──
        if len(self.plant_id_cache) > 0:
            print("=== plant_id_cache[0] ===")
            print(type(self.plant_id_cache[0]))
            print(self.plant_id_cache[0])
            print("=== plant_id_cache[1] ===")
            print(self.plant_id_cache[1])
        # ─────────────────────────────────────────────────
        if self.results_cache:
            # post_adapt = [t[-1].get('test_loss') for t in self.results_cache if t]
            # zero_shot  = [t[-1].get('zero_shot_loss') for t in self.results_cache if t]
            post_adapt = [t[-1].get('test_loss_masked') for t in self.results_cache if t]
            zero_shot  = [t[-1].get('zero_shot_loss_masked') for t in self.results_cache if t]
            post_adapt = [l for l in post_adapt if l is not None]
            zero_shot  = [l for l in zero_shot  if l is not None]
            if post_adapt and zero_shot:
                gain = np.mean(zero_shot) - np.mean(post_adapt)
                print(
                    f"\n[META-TEST] zero-shot  = {np.mean(zero_shot):.6f} ± {np.std(zero_shot):.6f}\n"
                    f"[META-TEST] post-adapt = {np.mean(post_adapt):.6f} ± {np.std(post_adapt):.6f}\n"
                    f"[META-TEST] adapt gain = {gain:+.6f}  "
                    f"({'✅ adaptation helps' if gain > 0 else '❌ adaptation hurts'})"
                )

        from utils.ks_analysis import KSDistributionAnalyzer
        job_id    = os.environ.get("SLURM_JOB_ID", self.logger.version)
        output_dir = Path("output") / f"pv_timeseries_tft_v2/job_{job_id}"

        analyzer = KSDistributionAnalyzer(
            pred_cache     = self.pred_cache,
            support_cache  = self.support_cache,
            plant_id_cache = self.plant_id_cache,
            threshold      = self.zero_target_threshold,
            quantile_idx   = (0, 1, 2),   # q10, q50, q90 # CAREFUL!!!
            output_dir      = output_dir,
        )

        per_plant_df, aggregate_df = analyzer.run()

        self.log("test/ks_A_mean",     float(per_plant_df["ks_A_stat"].mean()))
        self.log("test/ks_B_mean",     float(per_plant_df["ks_B_stat"].mean()))
        self.log("test/coverage_mean", float(per_plant_df["coverage_pct"].mean()))

        # ── Per-Plant Tabelle ─────────────────────────────────────────────────
        if getattr(self, "_per_plant_test_cache", None):

            df = pd.DataFrame(self._per_plant_test_cache)
            df = df.dropna(subset=["zero_shot_MAE", "adapted_MAE"])
            df["gain_abs"] = df["zero_shot_MAE"] - df["adapted_MAE"]
            df["gain_pct"] = (df["gain_abs"] / df["zero_shot_MAE"].replace(0, float("nan"))) * 100
            df = df.sort_values("gain_abs", ascending=False).reset_index(drop=True)

            # ── Konsole ───────────────────────────────────────────────────────
            print("\n" + "═" * 80)
            print("  📊 Per-Plant Test Results")
            print("═" * 80)
            print(df.to_string(
                index=True,
                float_format=lambda x: f"{x:.4f}",
            ))
            print(f"\n  Ø Zero-Shot MAE : {df['zero_shot_MAE'].mean():.4f}")
            print(f"  Ø Adapted  MAE  : {df['adapted_MAE'].mean():.4f}")
            print(f"  Ø Gain          : {df['gain_abs'].mean():+.4f}")
            print("═" * 80)

            # ── CSV Export ────────────────────────────────────────────────────
            job_id = os.environ.get("SLURM_JOB_ID", getattr(self.logger, "version", "local"))
            out_dir = Path("output") / f"pv_timeseries_tft_v2/job_{job_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "per_plant_test_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  💾 Saved → {csv_path}")

    def _classify_param(self, param_name: str) -> str:
        """
        Maps parameter name to a carse model-component label
        (only used to group gradient/weight norms for loggings)"""
        if "vsn_enc" in param_name:
            return "vsn_enc"
        if "vsn_dec" in param_name:
            return "vsn_dec"
        if "_proj" in param_name:
            return "proj"
        if "encoder_grn" in param_name:
            return "encoder_grn"
        if "encoder" in param_name:
            return "encoder_lstm"
        if "post_attn_grn" in param_name or "attn_norm" in param_name:
            return "post_attn"
        if "attn" in param_name:
            return "attention"
        if param_name.startswith("head"):
            return "head"
        return "other"

    def on_before_optimizer_step(self, optimizer):
        """Lightning hook: computes and logs global and per-component gradient/weight
        norms (and their ratio) just before each optimizer step; optionally logs
        parameter/gradient histograms every `grad_hist_every_n_steps` steps."""
        # log only every N steps
        if self.global_step == 0:
            return
        if self.global_step % self.grad_log_every_n_steps != 0:
            return

        comp_grad_sq = defaultdict(float)
        comp_weight_sq = defaultdict(float)

        global_grad_sq = 0.0
        global_weight_sq = 0.0
        dead_params = 0
        grad_params = 0

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            comp = self._classify_param(name)

            # weight norm
            w_norm = p.data.detach().norm(2).item()
            comp_weight_sq[comp] += w_norm ** 2
            global_weight_sq += w_norm ** 2

            # grad norm
            if p.grad is None:
                continue

            g = p.grad.detach()
            g_norm = g.norm(2).item()

            comp_grad_sq[comp] += g_norm ** 2
            global_grad_sq += g_norm ** 2
            grad_params += 1

            if g_norm < 1e-6:
                dead_params += 1

        global_grad = math.sqrt(global_grad_sq)
        global_weight = math.sqrt(global_weight_sq)
        global_ratio = global_grad / (global_weight + 1e-12)

        self.log("grad/global_norm", global_grad, on_step=True, on_epoch=True, logger=True)
        self.log("weight/global_norm", global_weight, on_step=True, on_epoch=True, logger=True)
        self.log("grad_ratio/global", global_ratio, on_step=True, on_epoch=True, logger=True)

        if grad_params > 0:
            self.log(
                "grad/dead_param_fraction",
                dead_params / grad_params,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        for comp in sorted(set(comp_weight_sq) | set(comp_grad_sq)):
            g_norm = math.sqrt(comp_grad_sq.get(comp, 0.0))
            w_norm = math.sqrt(comp_weight_sq.get(comp, 0.0))
            ratio = g_norm / (w_norm + 1e-12)

            self.log(f"grad/{comp}_norm", g_norm, on_step=True, on_epoch=True, logger=True)
            self.log(f"weight/{comp}_norm", w_norm, on_step=True, on_epoch=True, logger=True)
            self.log(f"grad_ratio/{comp}", ratio, on_step=True, on_epoch=True, logger=True)

        # optional histograms, to see the evolution. Only used while debugging to avoid gradient vanishing
        if self.log_grad_histograms and self.global_step % self.grad_hist_every_n_steps == 0:
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_histogram"):
                for name, p in self.named_parameters():
                    if not p.requires_grad:
                        continue

                    exp.add_histogram(
                        f"weights/{name}",
                        p.detach().float().cpu(),
                        global_step=self.global_step,
                    )

                    if p.grad is not None:
                        exp.add_histogram(
                            f"grads/{name}",
                            p.grad.detach().float().cpu(),
                            global_step=self.global_step,
                        )
    def _support_is_empty(self) -> bool:        
        """Returns True if the current task's support set (self.X_train)
        contains zero samples (used to skip adaptation, e.g. for zero-shot)."""
        if isinstance(self.X_train, dict):
            return self.X_train["encoder_lengths"].size(0) == 0
        elif isinstance(self.X_train, tuple):
            return self.X_train[0].shape[0] == 0
        elif isinstance(self.X_train, torch.Tensor):
            return self.X_train.shape[0] == 0
        if isinstance(self.X_train, list):
            return len(self.X_train) == 0
        else:
            return self.X_train.shape[0] == 0

    def _adapt_per_group(self, loss, inner_model):
        """Applies per-group meta-learnable LRs via manual gradient step on l2l clone."""
        create = (not self.first_order) and state.train
        grads = torch.autograd.grad(
            loss,
            inner_model.parameters(),
            create_graph= create,
            retain_graph= create,
            allow_unused=True,
        )

        for (name, param), grad in zip(inner_model.named_parameters(), grads):
            if grad is None:
                param.update = None
                continue
            group_name = self.get_group_name_for_param(name)
            lr = self.meta_lrs.get_lr(group_name)
            param.update = -lr * grad

        l2l.update_module(inner_model)
        return inner_model

    def _capture_interpretation(self):
        """
        If the underlying model exposes TFT interpretation data
        (via _TFTFeatures.last_interpretation), capture it for
        post-hoc plotting. Only active during testing.

        This captures interpretation from the ACTUAL test tasks,
        not separate random forward passes.
        """
        # For MAML: self.inner_model wraps the full model
        # For ANIL: self.features is the body
        features = getattr(self, 'features', None)
        if features is None:
            # MAML case: unwrap inner_model → meta_model → model
            model = self.inner_model
            # l2l MAML clone wraps .module
            raw = getattr(model, 'module', model)
            features = getattr(raw, 'features', None)
            # PlainTFT uses _features property
            if features is None:
                features = getattr(raw, '_features', None)

        if features is not None and hasattr(features, 'last_interpretation'):
            interp = features.last_interpretation
            if interp is not None:
                self.interpretation_cache.append(interp)

    def _slice_samples(self, all_samples, k: int):
        """Slice first k samples from batched support data."""
        if k == 0:
            if isinstance(all_samples, dict):
                return {key: v[:0] if isinstance(v, torch.Tensor) else v
                        for key, v in all_samples.items()}
            elif isinstance(all_samples, tuple):
                return tuple(x[:0] for x in all_samples)
            else:
                return all_samples[:0]

        if isinstance(all_samples, dict):
            return {key: v[:k] if isinstance(v, torch.Tensor) else v
                    for key, v in all_samples.items()}
        elif isinstance(all_samples, tuple):
            return tuple(x[:k] for x in all_samples)
        else:
            return all_samples[:k]

    def _get_eval_k_values(self, n_samples: int) -> list:
        """Log-spaced k values for sample efficiency (avoids N full evals)."""
        if n_samples <= 30:
            return list(range(1, n_samples + 1))
        k_set = set([1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 75, 100])
        k_set.update(range(0, n_samples + 1, max(1, n_samples // 20)))


        return sorted(k for k in k_set if k < n_samples)

    def get_group_name_for_param(self, param_name: str) -> str:
        """Map a full parameter name to its learning rate group."""
        # Example: match the first sub-module name
        for group_name in self.meta_lrs.lrs.keys():
            if group_name in param_name:
                return group_name
        # Fallback to first group or raise
        return list(self.meta_lrs.lrs.keys())[-1]
    
    @staticmethod
    def _masked_error(preds, y, threshold, median_idx=1):
        """Report-only MAE on the median channel, excluding near-zero targets.
        Fully detached — cannot touch the meta-graph."""
        p = preds[0] if isinstance(preds, (tuple, list)) else preds
        p, yt = p.detach(), y.detach()

        # ── Quantile output: pick the median channel ──────────────
        if p.dim() == yt.dim() and p.shape[-1] != yt.shape[-1]:
            p = p[..., median_idx:median_idx + 1]   # [B, T, 1]

        mask = yt > threshold
        if mask.sum() == 0:                          # pure-night window guard
            return float("nan")
        return (p[mask] - yt[mask]).abs().mean().item()


    def _is_testing(self) -> bool:
        """True only during trainer.test(), not in the validation."""
        return bool(getattr(self.trainer, "testing", False))
    
    
''' This is a class for learnable learning rates, 
whcih is not yet used within the above module and hence not properly documented.'''
class MetaLearningRates(torch.nn.Module):
    def __init__(self, lr_dict: dict, device=None):
        super().__init__()
        # lr_dict: z.B. {"encoder": 1e-2, "classifier": 1e-1}
        self.lrs = torch.nn.ParameterDict()
        for group_name, init_lr in lr_dict.items():
            self.lrs[group_name] = torch.nn.Parameter(
                torch.tensor(float(init_lr), dtype=torch.float32)
            )
        #if device is not None:
         #   self.to(device)

    def get_lr(self, group_name: str) -> torch.Tensor:
        return self.lrs[group_name]
