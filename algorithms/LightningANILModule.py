''' ANIL pipeline for pytorch lightning.'''
from typing import TYPE_CHECKING

from typing_extensions import override

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import learn2learn as l2l

from algorithms.LightningMAMLModule import LightningMAMLModule
from algorithms.PipelineState import PipelineState as state

import logging
log = logging.getLogger("meta-netze")

class LightningANILModule(LightningMAMLModule):
    ''' Pytorch Lightning module for the ANIL algorithm.
    It overwrites init_algorithm so that there is a backbone named features and a head.
    Only the model head is adapted in inner learning. This is ensured in determine_inner_model function.
    Optim config function is also changed to contain all model parameters of features and head. This is neccessary due to changes in init_algorithm.
    Note, that this version does not support usage of higher (multi-lr-optim), because it was not yet needed to have more than one head block,
    and we did not yet modify meta-optimizers to have multiple learning rates.
    '''
    @override
    def init_algorithm(self):
        """ Initializes the meta-learning algorithm.
        This includes model wraping.
        """
        # Separate between the meta_model features and head.
        # Todo: Some proper logging in which one can see the model parts.
        self.features = self.meta_model.features
        self.head = self.meta_model.head
        # Apply MAML only to the head
        base_lr = self.fast_lr if isinstance(self.fast_lr, (int, float)) else 1e-3
        self.head = l2l.algorithms.MAML(
            self.head, # only head gets wrapper
            lr=base_lr, 
            first_order = self.first_order)

    @override
    def configure_optimizers(self):
        """Builds a single Adam meta-optimizer over both `features` and `head`
        parameters. If `fast_lr` is a dict, additionally registers 
        as an extra parameter group on the same optimizer.
        """
        all_parameters = list(self.features.parameters()) + list(self.head.parameters())
        self.meta_optimizer = optim.Adam(
            all_parameters,
            lr=self.meta_lr,
            weight_decay=self.weight_decay)
        
        # ── Per-group inner LRs (same logic as parent) ──────────────
        if isinstance(self.fast_lr, dict):
            from algorithms.LightningMAMLModule import MetaLearningRates
            self.meta_lrs = MetaLearningRates(self.fast_lr, device=self.device)
            base_lr = self.meta_lr if not isinstance(self.meta_lr, dict) \
                    else list(self.meta_lr.values())[0]
            self.meta_optimizer.add_param_group(
                {"params": self.meta_lrs.parameters(), "lr": base_lr})

        return self.meta_optimizer

    @override
    def determine_inner_model(self):
        ''' Determines the inner model, what simply is the full model in MAML.'''
        inner = self.head.clone()
        inner.train()
        return inner

    @override
    def unpack_and_encode_inputs(self, batch):
        """
        Baseline-Modus: batch = (X, y)          → len == 2
        Meta-Modus:     batch = (X_tr, y_tr, X_te, y_te) → len == 4
        """
        # ── Baseline path (len == 2) ─────────────────────────────────
        if len(batch) == 2:
            X, y = batch
            X = self.features(*X) if isinstance(X, (tuple, list)) \
                else self.features(X)
            return X, y

        # ── Meta path (len == 4) ─────────────────────────────────────
        X_train, y_train, X_test, y_test = batch

        if isinstance(X_train, dict):
            device = next(self.parameters()).device
            X_train = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in X_train.items()}
            X_test  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in X_test.items()}
            y_train = y_train.to(device)
            y_test  = y_test.to(device)
            with torch.no_grad():
                X_train = self.features(X_train)
            X_test = self.features(X_test)
            return X_train, y_train, X_test, y_test

        with torch.no_grad():
            X_train = self.features(*X_train) if isinstance(X_train, (tuple, list)) \
                    else self.features(X_train)

        X_test = self.features(*X_test) if isinstance(X_test, (tuple, list)) \
                else self.features(X_test)

        return X_train, y_train, X_test, y_test
    