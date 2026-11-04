''' Gradient-based pipeline class for pytorch lightning usage.'''

from typing import final

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
import torch 

from conf.configs import PipelineParams
from conf.lightning_configs import TrainerConfig
import logging

logger = logging.getLogger("meta-netze")


class LightningPipeline:
    ''' This is the overal main pipeline for any meta-learning using pytorch lightning.
    On the highest level it simply
    - Initializes objects like sampler, module etc.
    - provides meta-training and meta-testing functions, whcih simply utilize the pytorch_lightning trainer's fit and test methods.
    Note: meta-validation is explicitly called within meta-training, since rainer.fit receives a validation_dataloader from the module.

    Key notions:
    - module: The underyling pytoch module representing the algorithm and respective setups, e.g. LightningMAMLModule
    - sampler: The underyling task sampler, that initializes task pools, dataloaders and their underyling datasets.
        The datasets are pytorch lightning datasets, which must, however, function as the "task sampling" entity,
        i.e. a batch is a batch of tasks rather than a batch of samples (like normally).
        See SupervisedDataset in data folder for more details.
    '''
    @final
    def __init__(self, config: PipelineParams):
        ''' Initializes the meta-learning pipeline for Lightning MAML.'''
        self.config = PipelineParams.get_config()
        self.trainer_config = TrainerConfig.get_config()
        self.task_sampler = config.sampler
        self.module = config.module

        self.trainer = pl.Trainer(**self.trainer_config.__dict__)
        logger.debug("Successfully created trainer")

        self.eval = config.evaluations

    @final
    def meta_training(self):
        ''' Meta-training function for pytorch lightning.'''
        logger.info("Starting meta-training")
        self.trainer.fit(
            model=self.module,
            train_dataloaders=self.task_sampler.train_dataloader(),
            val_dataloaders=self.task_sampler.val_dataloader()
        )

    @final
    def meta_testing(self):
        ''' Manually calling meta-testing using the test pool provided by sampler.test_dataloader.'''
        torch.cuda.empty_cache()          # HEAD: free GPU before test run

        self.module.current_task = -1     # metaRL: reset task counter
        pl.seed_everything(42, workers=True)
        logger.info("Starting meta-testing")

        test_dataloader = self.task_sampler.test_dataloader()
        test_results = self.trainer.test(
            self.module,
            dataloaders=[test_dataloader]
        )

        logger.info("Starting evaluation of meta-testing results")
        self.eval(self.module.results_cache)

        # ── Build MetaPrediction for TFTTester ────────────────────
        # pred_cache stores correctly paired (predictions, targets)
        # from each test task. We build a synthetic DataFrame so that
        # MetaPrediction's sliding-window indexing aligns predictions
        # with their TRUE actuals (not another plant's data).
        from evaluation.meta_prediction import MetaPrediction

        pred_cache = self.module.pred_cache
        plant_id_cache = getattr(self.module, 'plant_id_cache', [])
        support_cache  = getattr(self.module, 'support_cache',  None)

        # time_cache = getattr(self.module, 'time_cache', None)
        # support_time_cache = getattr(self.module, "support_time_cache", None)
        # ── TEMP debug: inspect cache structure ───────────────
        print(f"[DEBUG] pred_cache len = {len(pred_cache)}")
        print(f"[DEBUG] plant_id_cache len = {len(plant_id_cache)}")
        if plant_id_cache:
            print(f"[DEBUG] plant_id_cache[0] type = {type(plant_id_cache[0])}")
            print(f"[DEBUG] plant_id_cache[0] = {plant_id_cache[0]}")
        if pred_cache:
            p, t = pred_cache[0]
            print(f"[DEBUG] preds[0].shape = {tuple(p.shape)}, "
                  f"y[0].shape = {tuple(t.shape)}")
        # ──────────────────────────────────────────────────────
        
        if pred_cache:
            all_preds   = torch.cat(
                [p for p, _ in pred_cache], dim=0)  # [N, look_ahead, out]
            all_targets = torch.cat(
                [t for _, t in pred_cache], dim=0)  # [N, look_ahead, 1]

            N, look_ahead, out_size = all_preds.shape
            targets   = [
                self.config.data_config.get('config', {}).get('target', 'active_power_kW')]
            time_col  = "step_idx"
            quantiles = self.config.loss_config.get(
                'config', {}
            ).get('quantiles', None) or [0.5]

            # ── Build synthetic test_df ───────────────────────────
            # MetaPrediction indexes: iloc[start_idx + i : start_idx + i + look_ahead]
            # So window i uses rows [i, i + look_ahead).
            # We need N + look_ahead - 1 rows total.
            # total_rows   = N + look_ahead - 1
            total_rows = N * look_ahead
            actuals_flat = np.full(total_rows, np.nan, dtype=np.float32)

            # Fill actuals: window i writes targets[i] into rows [i, i+look_ahead)
            # Overlapping regions get the LAST write (arbitrary but consistent)
            for i in range(N):
                tgt = all_targets[i, :, 0].numpy()  # [look_ahead]
                actuals_flat[i * look_ahead : (i+1) * look_ahead] = tgt

            df = pd.DataFrame({
                time_col: np.arange(total_rows),
                "active_power_kW": actuals_flat,
            })
            self.meta_prediction = MetaPrediction(
                predictions=all_preds,
                targets=targets,
                test_df=df,
                time_column_name=time_col,
                quantiles=quantiles,
                start_idx=0,
            )
            # ── Attach plant metadata for denormalization ─────────
            self.meta_prediction.plant_info = plant_id_cache
        else:
            self.meta_prediction = None


        from evaluation.PredictionPlotEvaluator import PredictionPlotEvaluator

        # print(type(time_cache[0]), np.asarray(time_cache[0]).shape, time_cache[0])

        time_cache         = getattr(self.module, 'time_cache', None)
        support_time_cache = getattr(self.module, "support_time_cache", None)

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        # plot_dir = os.path.join(self.config.plot_dir, f"job_{job_id}")
        if pred_cache:
            PredictionPlotEvaluator(
                pred_cache         = pred_cache,
                plant_id_cache     = plant_id_cache,
                output_dir         = str(self.config.output_folder) + f"/prediction_plots/job_{job_id}/",
                quantiles          = quantiles,
                select             = "cherry",
                n_each             = 10,
                support_cache      = support_cache,
                time_cache         = time_cache,           
                support_time_cache = support_time_cache,   
                flatten            = "average",
            ).run()