''' Generic wrapper: takes any LightningModule (subclass) and replaces ONLY training_step
    with an ordinary supervised training step.'''

import logging
import torch
import pytorch_lightning as pl

from algorithms.PipelineState import PipelineState as state

logger = logging.getLogger("meta-netze")

def _ordinary_training_step(self, batch, batch_idx):
    state.train = True
    logger.debug("Ordinary training step. Training status: %s", state.train)

    opt = self.optimizers()
    opt.zero_grad()

    model = self.unpack_model()
    head  = model.head if hasattr(model, "head") else model

    # ── ANIL has features-Extraktor, MAML does not ───────────────
    is_anil = hasattr(self, "features") and callable(getattr(self, "features"))

    if len(batch) == 4:
        X_train, y_train, X_test, y_test = batch

        n_tasks        = X_train[0].shape[0] if isinstance(X_train, (tuple, list)) else X_train.shape[0]
        n_tasks_to_use = min(n_tasks, 10000)
        task_indices   = torch.randperm(n_tasks)[:n_tasks_to_use]
        logger.debug("Using %d / %d tasks in this step", n_tasks_to_use, n_tasks)

        total_loss_value = 0.0
        for t_idx in task_indices:
            t_idx = t_idx.item()

            X_t = tuple(x[t_idx] for x in X_train) if isinstance(X_train, (tuple, list)) else X_train[t_idx]
            X_v = tuple(x[t_idx] for x in X_test)  if isinstance(X_test,  (tuple, list)) else X_test[t_idx]
            y_t = y_train[t_idx]
            y_v = y_test[t_idx]

            if is_anil:
                # ── ANIL: Backbone encoded, Head macht Vorhersage ──
                enc_t = self.features(*X_t) if isinstance(X_t, tuple) else self.features(X_t)
                enc_v = self.features(*X_v) if isinstance(X_v, tuple) else self.features(X_v)
                X = torch.cat([enc_t, enc_v], dim=0)
                y = torch.cat([y_t,   y_v],   dim=0)
                loss, _ = self.loss_fn(head, X, y)
            else:
                # ── MAML: complete model takes the raw input ──────────────
                X = tuple(torch.cat([a, b], dim=0) for a, b in zip(X_t, X_v)) \
                    if isinstance(X_t, tuple) \
                    else torch.cat([X_t, X_v], dim=0)
                y = torch.cat([y_t, y_v], dim=0)
                loss, _ = self.loss_fn(model, X, y)

            self.manual_backward(loss / n_tasks_to_use)
            total_loss_value += loss.detach()

        total_loss = total_loss_value / n_tasks_to_use

    else:
        X, y = batch
        if is_anil:
            X    = self.features(*X) if isinstance(X, (tuple, list)) else self.features(X)
            loss, _ = self.loss_fn(head, X, y)
        else:
            loss, _ = self.loss_fn(model, X, y)

        self.manual_backward(loss)
        total_loss = loss.detach()

    opt.step()

    self.log(
        "train_loss", total_loss,
        prog_bar=True, on_step=True, on_epoch=True,
        sync_dist=getattr(self, "data_parallel", False),
    )
    return total_loss

def _ordinary_on_fit_start(self):
    """Runs once before training. Builds the full-class training head
    and makes sure the (meta-)optimizer actually trains it, since baseline is a
    plain supervised loop."""
    if self.config.classification:
        self._cached_train_head = None
        self._reinit_head(self.sampler.train_dataset.unique_classes)

        # Baseline trains the head -> ensure the optimizer owns the fresh head params.
        model = self.unpack_model()
        head_params = list(model.last_layer.parameters())
        head_param_ids = {id(p) for p in head_params}
        # Drop any stale head group, then add the current one.
        self.meta_optimizer.param_groups = [
            g for g in self.meta_optimizer.param_groups
            if not any(id(p) in head_param_ids for p in g["params"])
        ]
        self.meta_optimizer.add_param_group({"params": head_params})

def _ordinary_on_validation_start(self):
    """Runs once per validation phase. Save the training head, swap in the N-way
    (e.g. 3-class) evaluation head used by the task structure."""
    if self.config.classification:
        self._cached_train_head = self.unpack_model().last_layer
        self._reinit_head(self.sampler.val_dataset.M_way)

def _ordinary_on_validation_end(self):
    """Restore the training head so the next training epoch resumes correctly."""
    if self.config.classification:
        if getattr(self, "_cached_train_head", None) is not None:
            self.unpack_model().last_layer = self._cached_train_head
            self._cached_train_head = None

def _ordinary_on_test_start(self):
    if self.config.classification:
        self._cached_train_head = getattr(self.unpack_model(), "last_layer", None)
        self._reinit_head(self.sampler.test_dataset.M_way)

def _ordinary_on_test_end(self):
    self.on_validation_end()

def make_ordinary_module(module_cls):
    ''' Factory: returns a subclass whose training_step runs an ordinary supervised
    loop, and which re-initializes the classification head once per
    training/validation/testing phase (train = full classes, eval = N-way).'''
    if not issubclass(module_cls, pl.LightningModule):
        raise TypeError(f"{module_cls} must be a LightningModule subclass")

    wrapped = type(
        f"Ordinary{module_cls.__name__}",
        (module_cls,),
        {
            "training_step": _ordinary_training_step,
            "on_fit_start": _ordinary_on_fit_start,
            "on_validation_start": _ordinary_on_validation_start,
            "on_validation_end": _ordinary_on_validation_end,
            "on_test_start": _ordinary_on_test_start,
            "on_test_end": _ordinary_on_test_end,
        },
    )
    return wrapped