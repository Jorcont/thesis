import yaml
import sys
import os
import traceback
from time import perf_counter
import pytorch_lightning as pl
from pathlib import Path
import torch
import numpy as np
import random

from conf.configs import PipelineParams
import logging
import logging
import torch
import torch.utils.data._utils.collate as _collate_module

logger = logging.getLogger("meta-netze")

_original_collate_tensor_fn = _collate_module.collate_tensor_fn


from utils.logging_uitls import setup_logger
logger = setup_logger(os.environ.get("LOG_LEVEL", "DEBUG"))

if __name__ == "__main__":
    # pl.seed_everything(42, workers=True)
    # print(sys.argv)

    # if len(sys.argv) != 2:
    #     raise ValueError("Configuration file not provided. "
    #                      "Call python full_pipeline.py <config_file>")

    # conf_file = sys.argv[1]

    # if not os.path.exists(conf_file):
    #     raise ValueError(f"Configuration file {conf_file} does not exist.")

    # with open(conf_file, 'r') as file:
    #     config = yaml.safe_load(file)
    # config = PipelineParams(config_path=conf_file, **config)
    # logger.info(f"Loaded config from file {conf_file}")
    # config.print_config()
    print(sys.argv)

    if len(sys.argv) != 2:
        raise ValueError("Configuration file not provided. "
                         "Call python main.py <config_file>")

    conf_file = sys.argv[1]
    if not os.path.exists(conf_file):
        raise ValueError(f"Configuration file {conf_file} does not exist.")

    with open(conf_file, 'r') as file:
        config = yaml.safe_load(file)

    config = PipelineParams(config_path=conf_file, **config)

    # ── Print config IMMEDIATELY after creation ──────────────────
    print(">>> print_config START")
    config.print_config()
    print(">>> print_config END")

    # ── Seed: read from YAML, fall back to 42 ────────────────────
    seed = getattr(config, "seed", 42)
    print(f">>> Random seed from YAML : {seed}")

    pl.seed_everything(seed, workers=True)   # ← use YAML seed, not hardcoded

    # Verify seed was applied
    print(f">>> Torch seed after pl.seed_everything : {torch.initial_seed()}")
    print(f">>> NumPy  seed after pl.seed_everything: {np.random.get_state()[1][0]}")
    print(f">>> Python seed after pl.seed_everything: {random.getstate()[1][0]}")

    start = perf_counter()
    pipeline = config.pipeline
    if config.meta_training_iter >0:
        pipeline.meta_training()
    pipeline.meta_testing()

    from conf.sampler_configs import SamplerConfig

    for split in ["train_dataset", "val_dataset", "test_dataset"]:
        ds = getattr(SamplerConfig.get_config().sampler, split, None)
        if ds is not None and hasattr(ds, "print_profiling"):
            print(f"\n[Profiling] {split}")
            ds.print_profiling()



    end = perf_counter()
    print("\nExecution terminated. Check the results and outputs\n")
    print("Elapsed time during the whole program in seconds:", end - start)

    job_id = os.environ.get("SLURM_JOB_ID", "local")


    try:
        from evaluation.TFTTester import TFTTester
        print("TFTTESTER IS RUNNING ...")
        if hasattr(pipeline, "meta_prediction") and pipeline.meta_prediction is not None:
            save_path = str(config.output_folder) + f"/plots/job_{job_id}/"
            os.makedirs(save_path, exist_ok=True)
            # Set fast_mode = False for final results
            TFTTester(save_path=save_path, pred=pipeline.meta_prediction, fast_mode=False)
            print(f"✅ TFTTester plots saved to {save_path}")
        else:
            print("⚠️  No meta_prediction found on pipeline — skipping TFTTester.")
            print(f"[DEBUG] pipeline attributes: {[a for a in dir(pipeline) if not a.startswith('_')]}")
    except Exception as e:
        print(f"⚠️  TFTTester failed: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ── Sample Efficiency Curve ───────────────────────────────────
    print(f"[DEBUG] About to run SampleEfficiencyPlotter")
    print(f"[DEBUG] results_cache length: {len(pipeline.module.results_cache)}")
    print(f"[DEBUG] first entry: {pipeline.module.results_cache[0][:2] if pipeline.module.results_cache else 'EMPTY'}")
    

    def _debug_results_health(results_cache):
        print("\n" + "="*60)
        print(f"[HEALTH] tasks total: {len(results_cache)}")

        k_seen, n_entries, none_k = set(), [], 0
        empty_test, nan_masked = 0, 0

        for task in results_cache:
            n_entries.append(len(task))
            for e in task:
                k = e.get("k_shot")
                if k is None:
                    none_k += 1
                else:
                    k_seen.add(k)
                if not e["losses"]["test"]:
                    empty_test += 1
                zm, tm = e.get("zero_shot_loss_masked"), e.get("test_loss_masked")
                if (zm is not None and np.isnan(zm)) or (tm is not None and np.isnan(tm)):
                    nan_masked += 1

        print(f"[HEALTH] entries/task  min={min(n_entries)} "
            f"max={max(n_entries)} mean={np.mean(n_entries):.1f}")
        print(f"[HEALTH] distinct k values: {sorted(k_seen)}")
        print(f"[HEALTH] entries with k_shot=None : {none_k}   ← should be 0")
        print(f"[HEALTH] entries with empty test  : {empty_test}")
        print(f"[HEALTH] entries with NaN masked  : {nan_masked}")

        # Sanity: does adapted beat zero-shot on average?
        z = [e["zero_shot_loss_masked"] for t in results_cache for e in t
            if e.get("zero_shot_loss_masked") is not None
            and not np.isnan(e["zero_shot_loss_masked"])]
        a = [e["test_loss_masked"] for t in results_cache for e in t
            if e.get("test_loss_masked") is not None
            and not np.isnan(e["test_loss_masked"])]
        if z and a:
            print(f"[HEALTH] mean zero-shot masked : {np.mean(z):.4f}")
            print(f"[HEALTH] mean adapted  masked : {np.mean(a):.4f}")
            print(f"[HEALTH] adaptation helps?     : {np.mean(a) < np.mean(z)}")
        print("="*60 + "\n")

    _debug_results_health(pipeline.module.results_cache)


    def _inspect_random_tasks(results_cache, n=12, seed=42, skip_nan_tasks=True):
        rng = random.Random(seed)

        # optional: NaN-Tasks überspringen, um "gesunde" Kurven zu sehen
        nan_tasks = set()
        for i, task in enumerate(results_cache):
            for e in task:
                zm, tm = e.get("zero_shot_loss_masked"), e.get("test_loss_masked")
                if (zm is not None and np.isnan(zm)) or (tm is not None and np.isnan(tm)):
                    nan_tasks.add(i); break

        pool = [i for i in range(len(results_cache))
                if not (skip_nan_tasks and i in nan_tasks)]
        picks = sorted(rng.sample(pool, min(n, len(pool))))

        print("\n" + "="*70)
        print(f"[INSPECT] sampled tasks: {picks}  (skip_nan={skip_nan_tasks})")
        print("="*70)

        for i in picks:
            task = results_cache[i]
            # nach k sortieren
            rows = sorted(task, key=lambda e: (e.get("k_shot") is None,
                                            e.get("k_shot", -1)))
            print(f"\n── Task {i} ── ({len(rows)} entries)")
            print(f"{'k':>5} | {'zero_masked':>12} | {'adapt_masked':>12} | "
                f"{'Δ (z-a)':>9} | {'test_mae[last]':>14}")
            print("-"*66)
            for e in rows:
                k  = e.get("k_shot")
                zm = e.get("zero_shot_loss_masked")
                tm = e.get("test_loss_masked")
                d  = (zm - tm) if (zm is not None and tm is not None
                                and not np.isnan(zm) and not np.isnan(tm)) else float("nan")
                last_mae = e["mae"]["test"][-1] if e["mae"]["test"] else float("nan")
                print(f"{str(k):>5} | {zm:12.5f} | {tm:12.5f} | "
                    f"{d:9.5f} | {last_mae:14.5f}")

            # Kurvenform: verbessert sich adapt_masked mit steigendem k?
            ks  = [e.get("k_shot") for e in rows if e.get("k_shot") is not None]
            tms = [e.get("test_loss_masked") for e in rows if e.get("k_shot") is not None]
            if len(ks) >= 3:
                first, last = tms[0], tms[-1]
                trend = "↓ improves" if last < first else "↑ worsens"
                print(f"      trend k={ks[0]}→{ks[-1]}: {first:.5f} → {last:.5f}  {trend}")
        print("="*70 + "\n")

    _inspect_random_tasks(pipeline.module.results_cache, n=12, seed=42, skip_nan_tasks=True)

    from collections import Counter
    nan_by_k = Counter()
    for task in pipeline.module.results_cache:
        for e in task:
            zm, tm = e.get("zero_shot_loss_masked"), e.get("test_loss_masked")
            if (zm is not None and np.isnan(zm)) or (tm is not None and np.isnan(tm)):
                nan_by_k[e.get("k_shot")] += 1
    print(f"[HEALTH] NaN masked by k_shot: {dict(sorted(nan_by_k.items(), key=lambda x:(x[0] is None, x[0])))}")


    nan_tasks = set()
    for i, task in enumerate(pipeline.module.results_cache):
        for e in task:
            zm, tm = e.get("zero_shot_loss_masked"), e.get("test_loss_masked")
            if (zm is not None and np.isnan(zm)) or (tm is not None and np.isnan(tm)):
                nan_tasks.add(i)
    print(f"[HEALTH] tasks with ANY NaN masked: {len(nan_tasks)} → {sorted(nan_tasks)}")

    try:
        from evaluation.SampleEfficiencyPlotter import SampleEfficiencyPlotter

        if hasattr(pipeline, 'module') and hasattr(pipeline.module, 'results_cache'):
            # save_path = str(config.output_folder) + "/plots/"
            save_path = str(config.output_folder) + f"/plots/job_{job_id}/"
            os.makedirs(save_path, exist_ok=True)

            data_cfg = config.data_config.get('config', {})
            lb = data_cfg.get('look_back_steps', 96)
            la = data_cfg.get('look_ahead_steps', 96)

            plotter = SampleEfficiencyPlotter(
                results_cache=pipeline.module.results_cache,
                save_path=save_path,
                look_back_steps=lb,
                look_ahead_steps=la,
                step_minutes=15.0,
                chunk_size=data_cfg.get("chunk_size", 1),
                window_stride = data_cfg.get("window_stride", 1)
            )
            plotter.plot()
    except Exception as e:
        print(f"⚠️  SampleEfficiencyPlotter failed: {type(e).__name__}: {e}")
        traceback.print_exc()
"""
    # ── TFT Interpretation Plots ──────────────────────────────────
    try:
        from evaluation.TFTInterpretation import TFTInterpretation

        module = pipeline.module if hasattr(pipeline, 'module') else None

        if module is not None and hasattr(module, 'interpretation_cache') \
                and len(module.interpretation_cache) > 0:

            save_path = str(config.output_folder) + "/interpretation/"
            os.makedirs(save_path, exist_ok=True)

            # Variable names from data config
            data_cfg = config.data_config.get('config', {})
            encoder_vars = data_cfg.get('time_varying_known_reals', []) + \
                           data_cfg.get('time_varying_unknown_reals', [])
            decoder_vars = data_cfg.get('time_varying_known_reals', [])
            static_vars  = ["group_id"]

            interp = TFTInterpretation(
                save_path=save_path,
                encoder_variables=encoder_vars,
                decoder_variables=decoder_vars,
                static_variables=static_vars,
            )

            # Feed cached interpretations from actual test tasks
            for interpretation in module.interpretation_cache:
                interp.accumulate(interpretation)

            interp.plot_all()
            print(f"✅ Interpretation plots saved to {save_path} "
                  f"({len(module.interpretation_cache)} tasks)")
        else:
            if module is not None:
                print("⚠️  No TFT interpretation data captured during testing "
                      "(model may not be PF_TFT).")
            else:
                print("⚠️  Could not access module for interpretation.")
    except Exception as e:
        print(f"⚠️  TFT Interpretation failed: {type(e).__name__}: {e}")
        traceback.print_exc()


    print("*************** JOB HAS FINISHED ************************")

    # Todo: Save config in checkpoint_dir.
    # Append elapsed time in result.txt
    result_file = str(config.output_folder) + "/result.txt"
    # Todo: For some reason, this file does not get created.
    #   It worked before changing the path to the inner folder structure of output folder.
    with open(result_file, "a") as result_file:
        device = config.device
        meta_training_iter = config.meta_training_iter
        devices = config.trainer_config["devices"]
        result_file.write(
            f"Settings: device={device}, "
            f"meta_training_iter={meta_training_iter}, "
            f"devices={devices}\n "
            f"Elapsed time: {end - start} seconds\n\n")
"""
