"""
aggregate_results.py

Combines:
  (A) Per-plant MAE aggregation from `per_plant_test_results.csv`
      -> feeds Table "aggregate_stats" and the MAE columns of
         Table "aggregate_performance".
  (B) Adaptation-speed aggregation from the SLURM `.err` logs via
      `utils/analyze_log.py` (same job_ids / seeds as (A))
      -> feeds a future "sample efficiency / adaptation speed" table
         and the per-step adaptation curve (mean +/- std across seeds),
         PLUS an un-normalized adaptation-speed metric (trapezoidal AUC
         of the raw support loss curve, see compute_auc_per_task).
  (C) Adaptation gain aggregation from `per_plant_test_results.csv`
      -> feeds Table "gain_stats" (mean/median/std of gain_abs, mean
         gain_pct, share of near-zero gain, share of negative gain,
         and the mean-minus-median gap) and the gain violin figure
         (per-plant gain_abs distribution per config, with mean and
         median explicitly marked).
  (D) Sample-efficiency aggregation (MAE / masked loss vs. k-shot).
  (E) Final test-set distribution/calibration metrics (Test A, Test B,
      Coverage) from the SLURM `.out` logs.
  (F) Best-/worst-case plant tables (Section 6.4): easiest/hardest
      zero-shot, best absolute/relative gain, worst (negative) gain.
  (G) Case-study forecast curves (Section 6.5) from pre-generated
      Plotly HTML (PredictionPlotEvaluator output).
  (H) Cross-configuration paired comparison (Section 6.7): Wilcoxon
      signed-rank test on common plants between configuration pairs.
  (I) Lightning TensorBoard training-curve aggregation (meta_train/
      val/test loss, zero-shot loss, masked + unmasked, epoch-level)
      via `utils/read_lightning_logs.py`, from `lightning_logs/
      version_<job_id>` -> per-config CSV + figure, plus a
      cross-configuration comparison figure per metric.
    (J) Cherry-pick 3×3 grid (Section 6.4 companion): hardest/easiest start,
      best abs/rel gain, best adapted MAE, best scale-free perf., worst
      negative gain, largest/smallest rated power. Built from the
      SEED-AVERAGED per-plant table `plant_tables[config_id]`
      -> per-config PNG under aggregate_output/cherry_pick_plots/.

All parts share the same CONFIGS dict (job_ids = seeds).

Run:
    python aggregate_results.py            # all configs in CONFIGS
    python aggregate_results.py C1         # only C1
    python aggregate_results.py C1 C4      # only C1 and C4
"""

import os
import re
from typing import Dict, List, Optional

import sys
import time
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import json

# ----------------------------------------------------------------------
# 0) Load utils/analyze_log.py as a module (reuse it, no new file)
# ----------------------------------------------------------------------
_ANALYZE_LOG_PATH = Path(__file__).resolve().parent / "utils" / "analyze_log.py"


def _load_analyze_log():
    spec = importlib.util.spec_from_file_location("analyze_log", _ANALYZE_LOG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze_log = _load_analyze_log()

# ----------------------------------------------------------------------
# 0b) Load utils/read_lightning_logs.py as a module (reuse it for Part I)
# ----------------------------------------------------------------------
_READ_LOGS_PATH = Path(__file__).resolve().parent / "utils" / "read_lightning_logs.py"


# ----------------------------------------------------------------------
# 0c) Load utils/cherry_pick_plots.py as a module (reuse it for Part J)
# ----------------------------------------------------------------------
_CHERRY_PICK_PATH = Path(__file__).resolve().parent / "utils" / "cherry_pick_plots2.py"


def _load_cherry_pick_plots():
    spec = importlib.util.spec_from_file_location("cherry_pick_plots", _CHERRY_PICK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cherry_pick_plots = _load_cherry_pick_plots()

def _load_read_lightning_logs():
    spec = importlib.util.spec_from_file_location("read_lightning_logs", _READ_LOGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


read_lightning_logs = _load_read_lightning_logs()

# ----------------------------------------------------------------------
# 1) EDIT THIS: configurations, abbreviations, job IDs (= seeds)
# ----------------------------------------------------------------------
BASE_DIR = Path("output/pv_timeseries_tft_v2")
LOG_DIR = Path("output/logs")
CSV_NAME = "per_plant_test_results.csv"
CONVERGENCE_THRESHOLD = 0.9  # "steps to reach X% of total support-loss improvement"
NEAR_ZERO_GAIN_THRESHOLD = 0.001  # for %(|gain_abs| < threshold) in gain_stats

CONFIGS = {
    "C1": {"abbrev": "LSTM-Base",   "job_ids": ["313107", "325690", "339525", "360099", "361960"]},
    "C2": {"abbrev": "TFT-Base",    "job_ids": ["313111", "325689", "338032", "346439"]},
    "C3": {"abbrev": "LSTM-FOMAML", "job_ids": ["312857", "323633", "330432", "339537", "355278"]},
    "C4": {"abbrev": "TFT-FOMAML",  "job_ids": ["325688", "324498", "327175", "335118", "346427"]},
    "C5": {"abbrev": "LSTM-ANIL",   "job_ids": ["312856", "323634", "335036", "342894", "356336"]},
    "C6": {"abbrev": "TFT-ANIL",    "job_ids": ["335162", "312851", "322436", "327176", "346443"]},
    "C7": {"abbrev": "LSTM-NIL",    "job_ids": ["312859", "324287", "342953", "356337"]},
    "C8": {"abbrev": "TFT-NIL",     "job_ids": ["312852", "323632", "351537", "353948"]},
}

NO_INNER_LOOP_CONFIGS = {"C7", "C8"}
# ----------------------------------------------------------------------
# Optional CLI filter: restrict to a subset of configs, e.g.
#   python aggregate_results.py C1
#   python aggregate_results.py C1 C4
# If no args given, runs everything in CONFIGS.
# ----------------------------------------------------------------------
if len(sys.argv) > 1:
    requested = sys.argv[1:]
    unknown = [c for c in requested if c not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown config id(s): {unknown}. Available: {list(CONFIGS.keys())}")
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in requested}
    print(f"[info] Running only: {list(CONFIGS.keys())}")

OUT_DIR = Path("/mnt/data/kes/data/jdomingo/personal-new/thesis/aggregate_output")
OUT_DIR.mkdir(exist_ok=True)
SPEED_DIR = OUT_DIR / "adaptation_speed"
SPEED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Part D config: Sample efficiency (support-window ablation)
# ----------------------------------------------------------------------
SAMPLE_EFF_SEARCH_ROOT = Path("output")  # only this level is scanned, see below
SAMPLE_EFF_SUBDIR = "plots"
SAMPLE_EFF_FILES = {
    "test_mae_final": "sample_efficiency_stats_test_mae_final.csv",
    "test_loss_masked": "sample_efficiency_stats_test_loss_masked.csv",
}
SAMPLE_EFF_LABELS = {
    "test_mae_final": "Test MAE",
    "test_loss_masked": "Test loss (masked)",
}
# k_shot checkpoints for the condensed LaTeX summary table; checkpoints
# missing for a given config/seed grid are simply reported as "--".
SAMPLE_EFF_CHECKPOINTS = [0, 1, 2, 5, 10, 20]

SAMPLE_EFF_DIR = OUT_DIR / "sample_efficiency"
SAMPLE_EFF_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Part E config: Final test-set distribution/calibration metrics
#   (ks_A_mean, ks_B_mean, coverage_mean -- from the Lightning "Test
#    metric" table printed once at the very end of each SLURM .out log).
#   Read via tail-seek, NOT a full-file read, since these .out files can
#   be huge due to per-batch tqdm progress-bar output.
# ----------------------------------------------------------------------
OUT_LOG_ROOT = Path("output/logs")
OUT_LOG_FILE_GLOB_TMPL = "*_{job_id}.out"
OUT_LOG_TAIL_CHUNK = 65536            # initial bytes read from the end
OUT_LOG_TAIL_MAX_BYTES = 5_000_000    # safety cap (~5 MB) per file

FINAL_METRIC_KEYS = {
    "ks_A_mean": "test/ks_A_mean",
    "ks_B_mean": "test/ks_B_mean",
    "coverage_mean": "test/coverage_mean",
}

FINAL_METRICS_DIR = OUT_DIR / "final_test_metrics"
FINAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Part F config: Best-/worst-case plant tables
# ----------------------------------------------------------------------
BEST_WORST_N = 10
BEST_WORST_DIR = OUT_DIR / "best_worst_case"
BEST_WORST_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Part H config: Cross-configuration paired comparison
# ----------------------------------------------------------------------
CROSS_CONFIG_PAIRS = {
    # Section 6.7.2 — Backbone comparison (same algorithm, LSTM vs TFT)
    "backbone": [("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("C7", "C8")],
    # Section 6.7.1 — Meta-learning vs. non-meta-learning (same backbone)
    "meta_vs_nonmeta": [("C1", "C3"), ("C1", "C5"), ("C1", "C7"),
                         ("C2", "C4"), ("C2", "C6"), ("C2", "C8")],
    # Section 6.7.3 — Inner-loop variants (same backbone)
    "inner_loop_variants": [("C3", "C5"), ("C3", "C7"), ("C5", "C7"),
                             ("C4", "C6"), ("C4", "C8"), ("C6", "C8")],
}

CROSS_CONFIG_DIR = OUT_DIR / "cross_config"
CROSS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Part G config: Case-study forecast curves (Section 6.5) from the
# pre-generated Plotly HTML produced by PredictionPlotEvaluator.
#   output/<unknown_folder>/prediction_plots/job_<job_id>/
#   -- same one-level-scan pattern as Parts D/E (only "<unknown_folder>"
#      is variable; "prediction_plots/job_<id>/" is a fixed sub-path).
# ----------------------------------------------------------------------
import base64

PRED_PLOTS_SEARCH_ROOT = Path("output")
PRED_PLOTS_SUBDIR = "prediction_plots"
PRED_PLOTS_SUMMARY_NAME = "prediction_summary.csv"

CASE_STUDY_DIR = OUT_DIR / "case_studies"
CASE_STUDY_DIR.mkdir(parents=True, exist_ok=True)

_PLOTLY_DTYPE_MAP = {
    "f8": np.float64, "f4": np.float32,
    "i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64,
    "u1": np.uint8, "u2": np.uint16, "u4": np.uint32, "u8": np.uint64,
}

# ----------------------------------------------------------------------
# Part I config: Lightning TensorBoard log aggregation (training curves)
#   lightning_logs/version_<job_id> -- directly under project root.
# ----------------------------------------------------------------------
LIGHTNING_LOGS_ROOT = Path("lightning_logs")

# masked + unmasked epoch-level curves (friendly names match auto_rename()
# output in utils/read_lightning_logs.py)
LIGHTNING_CURVE_TAGS = {
    "meta_train_loss_epoch":            "meta_train_loss",
    "meta_train_loss_masked_epoch":     "meta_train_loss_masked",
    "meta_val_loss_epoch":              "meta_val_loss",
    "meta_val_loss_masked_epoch":       "meta_val_loss_masked",
    "meta_test_loss_epoch":             "meta_test_loss",
    "meta_test_loss_masked_epoch":      "meta_test_loss_masked",
    "zero_shot_val_loss_epoch":         "zero_shot_val_loss",
    "zero_shot_val_loss_masked_epoch":  "zero_shot_val_loss_masked",
    "zero_shot_test_loss_epoch":        "zero_shot_test_loss",
    "zero_shot_test_loss_masked_epoch": "zero_shot_test_loss_masked",
}
LIGHTNING_SCALAR_METRICS = {
    "meta_test_loss", "meta_test_loss_masked",
    "zero_shot_test_loss", "zero_shot_test_loss_masked",
}
LIGHTNING_CURVE_METRICS = set(LIGHTNING_CURVE_TAGS.values()) - LIGHTNING_SCALAR_METRICS

LIGHTNING_METRIC_LABELS = {
    "meta_train_loss":            "Meta-Train loss",
    "meta_train_loss_masked":     "Meta-Train loss (masked)",
    "meta_val_loss":              "Meta-Val loss",
    "meta_val_loss_masked":       "Meta-Val loss (masked)",
    "meta_test_loss":             "Meta-Test loss",
    "meta_test_loss_masked":      "Meta-Test loss (masked)",
    "zero_shot_val_loss":         "Zero-Shot Val loss",
    "zero_shot_val_loss_masked":  "Zero-Shot Val loss (masked)",
    "zero_shot_test_loss":        "Zero-Shot Test loss",
    "zero_shot_test_loss_masked": "Zero-Shot Test loss (masked)",
}

LIGHTNING_DIR = OUT_DIR / "lightning_curves"
LIGHTNING_DIR.mkdir(parents=True, exist_ok=True)
LIGHTNING_FIGURES_PER_CONFIG_DIR = LIGHTNING_DIR / "per_config_figures"
LIGHTNING_FIGURES_PER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Part J config: Cherry-pick 3×3 grid (seed-averaged per-plant table)
# ----------------------------------------------------------------------
CHERRY_PICK_TOP_N = 10
CHERRY_PICK_DIR = OUT_DIR / "cherry_pick_plots"
CHERRY_PICK_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# PART A — Per-plant MAE aggregation
# ═══════════════════════════════════════════════════════════════════

def load_job_csv(job_id: str) -> pd.DataFrame:
    path = BASE_DIR / f"job_{job_id}" / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {"plant_id", "zero_shot_MAE", "adapted_MAE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def load_config_frames(job_ids):
    return [load_job_csv(jid) for jid in job_ids]


def per_plant_averaged_across_seeds(dfs) -> pd.DataFrame:
    merged = dfs[0][["plant_id", "zero_shot_MAE", "adapted_MAE"]].rename(
        columns={"zero_shot_MAE": "zero_shot_MAE_seed0", "adapted_MAE": "adapted_MAE_seed0"}
    )
    for i, df in enumerate(dfs[1:], start=1):
        tmp = df[["plant_id", "zero_shot_MAE", "adapted_MAE"]].rename(
            columns={"zero_shot_MAE": f"zero_shot_MAE_seed{i}", "adapted_MAE": f"adapted_MAE_seed{i}"}
        )
        merged = merged.merge(tmp, on="plant_id", how="inner")

    zs_cols = [c for c in merged.columns if c.startswith("zero_shot_MAE_seed")]
    ad_cols = [c for c in merged.columns if c.startswith("adapted_MAE_seed")]
    merged["zero_shot_MAE_mean"] = merged[zs_cols].mean(axis=1)
    merged["adapted_MAE_mean"] = merged[ad_cols].mean(axis=1)

    if len(merged) != len(dfs[0]):
        print(f"  [info] plant_id overlap across seeds: {len(dfs[0])} -> {len(merged)} common plants "
              f"after inner join (expected: meta-test task sampling can differ across seeds)")

    return merged


def distribution_stats(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "std": series.std(ddof=1),
    }


def seed_level_overall_mean(dfs, column: str) -> np.ndarray:
    return np.array([df[column].mean() for df in dfs])


# ═══════════════════════════════════════════════════════════════════
# PART B — Adaptation-speed aggregation (built on utils/analyze_log.py)
# ═══════════════════════════════════════════════════════════════════

def find_log_file(job_id: str) -> Path:
    matches = sorted(LOG_DIR.glob(f"*_{job_id}.err"))
    if not matches:
        raise FileNotFoundError(f"No .err log found for job_id={job_id} in {LOG_DIR}")
    if len(matches) > 1:
        print(f"  [warn] multiple logs match job_id={job_id}: "
              f"{[m.name for m in matches]} -> using {matches[0].name}")
    return matches[0]


def compute_steps_to_threshold_fast(step_df: pd.DataFrame, threshold: float = CONVERGENCE_THRESHOLD) -> pd.DataFrame:
    """
    Vectorized replacement for analyze_log.compute_steps_to_threshold.
    Avoids the per-task Python loop (slow for >10k tasks).
    Returns one row per task: task_global_id, task_local_id, n_total_steps,
    steps_to_<threshold>pct (NaN if the task never reached the threshold).
    """
    if step_df.empty:
        return pd.DataFrame()

    col_name = f"steps_to_{int(threshold*100)}pct"
    df_sorted = step_df.sort_values(["task_global_id", "step"])

    # one row per task: task_local_id, n_total_steps (first occurrence)
    task_meta = (
        df_sorted.groupby("task_global_id", as_index=False)
        .first()[["task_global_id", "task_local_id", "n_total_steps"]]
    )

    # first step where frac_converged >= threshold, per task (vectorized min)
    reached = df_sorted[df_sorted["frac_converged"] >= threshold]
    first_reached = (
        reached.groupby("task_global_id", as_index=False)["step"]
        .min()
        .rename(columns={"step": col_name})
    )

    result = task_meta.merge(first_reached, on="task_global_id", how="left")
    return result


# STEP_LOSS_COL = "loss"  # <-- adjust to the actual raw-loss column name
#                         #     in analyze_log.build_step_curve_df's output
#                         #     (the one plot_adaptation_curve_across_seeds
#                         #     already plots as "Support-set train loss")


# def compute_auc_per_task(step_df: pd.DataFrame, loss_col: str = STEP_LOSS_COL) -> pd.DataFrame:
#     """
#     Un-normalized adaptation-speed metric: trapezoidal AUC of the RAW
#     (not frac_converged-normalized) support-loss curve ell_p(s), per task,
#     normalized only by the task's own step count (NOT by its total
#     improvement -- unlike s_p^90 / f_p(s), Eq. frac_converged/steps_to_90).

#     Returns one row per task: task_global_id, task_local_id, n_total_steps,
#     auc_mean (time-averaged raw loss over the trajectory; lower = faster
#     AND deeper convergence, in the loss's own units).
#     """
#     if step_df.empty:
#         return pd.DataFrame()

#     df_sorted = step_df.sort_values(["task_global_id", "step"])

#     def _trapz_auc(g):
#         s = g["step"].to_numpy()
#         y = g[loss_col].to_numpy()
#         if len(s) < 2:
#             return pd.Series({"auc_raw": np.nan, "step_span": 0.0})
#         return pd.Series({
#             "auc_raw": np.trapz(y, x=s),
#             "step_span": float(s[-1] - s[0]),
#         })

#     auc_df = (
#         df_sorted.groupby("task_global_id")
#         .apply(_trapz_auc)
#         .reset_index()
#     )
#     task_meta = (
#         df_sorted.groupby("task_global_id", as_index=False)
#         .first()[["task_global_id", "task_local_id", "n_total_steps"]]
#     )
#     result = task_meta.merge(auc_df, on="task_global_id", how="left")
#     with np.errstate(divide="ignore", invalid="ignore"):
#         result["auc_mean"] = np.where(
#             result["step_span"] > 0, result["auc_raw"] / result["step_span"], np.nan
#         )
#     return result

# Known non-loss / id columns that compute_steps_to_threshold_fast already
# relies on existing in step_df (task_global_id, task_local_id, n_total_steps,
# step, frac_converged). Anything else is a candidate for "the raw loss".
_STEP_DF_KNOWN_NON_LOSS_COLS = {
    "task_global_id", "task_local_id", "n_total_steps", "step", "frac_converged",
}

# Set this explicitly once you know the real column name (see the printed
# warning on first use), to skip auto-detection:
STEP_LOSS_COL: Optional[str] = None


def _infer_loss_column(step_df: pd.DataFrame) -> str:
    """
    Auto-detects the raw support-loss column in step_df produced by
    analyze_log.build_step_curve_df(), since its exact name is not
    guaranteed (it varies with the log-parsing implementation).
    """
    candidates = [
        "loss", "value", "support_loss", "train_loss",
        "support_train_loss", "raw_loss", "loss_value", "y",
    ]
    for c in candidates:
        if c in step_df.columns:
            return c

    remaining = [c for c in step_df.columns if c not in _STEP_DF_KNOWN_NON_LOSS_COLS]
    if len(remaining) == 1:
        return remaining[0]

    raise KeyError(
        f"Could not auto-detect the raw support-loss column in step_df. "
        f"Available columns: {list(step_df.columns)}. "
        f"Set STEP_LOSS_COL explicitly to the correct one of these at the "
        f"top of the script, then re-run."
    )


_loss_col_resolved: Optional[str] = None  # cache so we only auto-detect/print once


def compute_auc_per_task(step_df: pd.DataFrame, loss_col: Optional[str] = STEP_LOSS_COL) -> pd.DataFrame:
    """
    Un-normalized adaptation-speed metric: trapezoidal AUC of the RAW
    (not frac_converged-normalized) support-loss curve ell_p(s), per task,
    normalized only by the task's own step count (NOT by its total
    improvement -- unlike s_p^90 / f_p(s), Eq. frac_converged/steps_to_90).

    Returns one row per task: task_global_id, task_local_id, n_total_steps,
    auc_mean (time-averaged raw loss over the trajectory; lower = faster
    AND deeper convergence, in the loss's own units).
    """
    global _loss_col_resolved

    if step_df.empty:
        return pd.DataFrame()

    if loss_col is None or loss_col not in step_df.columns:
        if _loss_col_resolved is None or _loss_col_resolved not in step_df.columns:
            _loss_col_resolved = _infer_loss_column(step_df)
            print(f"  [info] AUC: auto-detected raw-loss column = '{_loss_col_resolved}' "
                  f"(available columns were: {list(step_df.columns)}). "
                  f"If this is wrong, set STEP_LOSS_COL explicitly.")
        loss_col = _loss_col_resolved

    df_sorted = step_df.sort_values(["task_global_id", "step"])

    def _trapz_auc(g):
        s = g["step"].to_numpy()
        y = g[loss_col].to_numpy()
        if len(s) < 2:
            return pd.Series({"auc_raw": np.nan, "step_span": 0.0})
        return pd.Series({
            "auc_raw": np.trapz(y, x=s),
            "step_span": float(s[-1] - s[0]),
        })

    auc_df = (
        df_sorted.groupby("task_global_id")
        .apply(_trapz_auc)
        .reset_index()
    )
    task_meta = (
        df_sorted.groupby("task_global_id", as_index=False)
        .first()[["task_global_id", "task_local_id", "n_total_steps"]]
    )
    result = task_meta.merge(auc_df, on="task_global_id", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        result["auc_mean"] = np.where(
            result["step_span"] > 0, result["auc_raw"] / result["step_span"], np.nan
        )
    return result

def combine_auc_across_seeds(auc_dfs: list) -> pd.Series:
    """
    Pools per-task auc_mean values across seeds (NO per-task join), same
    rationale as combine_steps_to_threshold_across_seeds: task identity is
    not required to match across seeds, and this avoids a combinatorial
    merge blow-up. Gives the ACROSS-TASK distribution of auc_mean, with
    seeds pooled (see Statistical Aggregation Convention, Section
    results_agg_convention) -- i.e. cross-task heterogeneity, not
    cross-seed variability.
    """
    valid = [d["auc_mean"].dropna() for d in auc_dfs if not d.empty]
    if not valid:
        return pd.Series(dtype=float)
    return pd.concat(valid, ignore_index=True)


def compute_adaptation_speed_for_job(job_id: str, threshold: float = CONVERGENCE_THRESHOLD):
    """Parse one .err log; return (step_df, step_summary, steps_to_threshold_df, auc_df)."""
    log_path = find_log_file(job_id)
    df, tasks = analyze_log.parse_log_file(str(log_path))
    step_df = analyze_log.build_step_curve_df(tasks)
    if step_df.empty:
        return step_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    step_summary = analyze_log.compute_step_summary(step_df)
    steps_to_thr = compute_steps_to_threshold_fast(step_df, threshold=threshold)
    auc_df = compute_auc_per_task(step_df)
    return step_df, step_summary, steps_to_thr, auc_df


def aggregate_step_curve_across_seeds(step_summaries: list) -> pd.DataFrame:
    """
    Combine per-seed step summaries (each already averaged over tasks within
    that seed) into mean +/- std ACROSS SEEDS, per adaptation step.
    """
    valid = [s for s in step_summaries if not s.empty]
    if not valid:
        return pd.DataFrame()

    merged = valid[0][["step", "mean"]].rename(columns={"mean": "mean_seed0"})
    for i, s in enumerate(valid[1:], start=1):
        tmp = s[["step", "mean"]].rename(columns={"mean": f"mean_seed{i}"})
        merged = merged.merge(tmp, on="step", how="inner")

    seed_cols = [c for c in merged.columns if c.startswith("mean_seed")]
    merged["mean_across_seeds"] = merged[seed_cols].mean(axis=1)
    merged["std_across_seeds"] = (
        merged[seed_cols].std(axis=1, ddof=1) if len(seed_cols) > 1 else np.nan
    )
    return merged


def combine_steps_to_threshold_across_seeds(steps_dfs: list, threshold_col: str) -> pd.Series:
    """
    Concatenate per-task convergence-step values across seeds (NO per-task
    join). Task identity does not need to match across seeds -- consistent
    with stochastic meta-learning task sampling, and avoids the quadratic
    merge blow-up (and OOM) caused by joining on a non-unique key
    (task_local_id repeats every time the same plant is sampled across
    training epochs, so a naive merge on it explodes combinatorially).
    """
    valid = [d[threshold_col].dropna() for d in steps_dfs if not d.empty and threshold_col in d.columns]
    if not valid:
        return pd.Series(dtype=float)
    return pd.concat(valid, ignore_index=True)


def plot_adaptation_curve_across_seeds(curve_df: pd.DataFrame, title: str, out_path: Path):
    if curve_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_df["step"], curve_df["mean_across_seeds"],
             marker="o", color="tab:blue", label="mean (across seeds)")
    if curve_df["std_across_seeds"].notna().any():
        ax.fill_between(
            curve_df["step"],
            curve_df["mean_across_seeds"] - curve_df["std_across_seeds"],
            curve_df["mean_across_seeds"] + curve_df["std_across_seeds"],
            alpha=0.2, color="tab:blue", label="std (across seeds)",
        )
    ax.set_xlabel("Adaptation step")
    ax.set_ylabel("Support-set train loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregate_box(per_plant_data: dict, out_path: Path):
    """
    per_plant_data: dict config_id -> {"zero_shot": np.array, "adapted": np.array, "abbrev": str}
    Produces one figure with grouped box plots: for each config, zero-shot and
    adapted side by side.
    """
    config_ids = list(per_plant_data.keys())
    n = len(config_ids)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(1.4 * n + 2, 5))

    positions_zs = np.arange(n) * 2.2
    positions_ad = positions_zs + 0.9

    data_zs = [per_plant_data[c]["zero_shot"] for c in config_ids]
    data_ad = [per_plant_data[c]["adapted"] for c in config_ids]
    labels = [per_plant_data[c]["abbrev"] for c in config_ids]

    bp_zs = ax.boxplot(data_zs, positions=positions_zs, widths=0.7,
                        patch_artist=True, showfliers=False)
    bp_ad = ax.boxplot(data_ad, positions=positions_ad, widths=0.7,
                        patch_artist=True, showfliers=False)

    for box in bp_zs["boxes"]:
        box.set(facecolor="tab:orange", alpha=0.6)
    for box in bp_ad["boxes"]:
        box.set(facecolor="tab:blue", alpha=0.6)

    tick_positions = (positions_zs + positions_ad) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("MAE (per plant)", fontsize=16)
    ax.set_title("Per-plant MAE distribution: zero-shot vs. adapted", fontsize=18)

    ax.legend(
        [bp_zs["boxes"][0], bp_ad["boxes"][0]],
        ["Zero-Shot", "Adapted"],
        loc="upper right",
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════
# PART C — Adaptation gain aggregation (gain_abs / gain_pct per plant)
# ═══════════════════════════════════════════════════════════════════

def compute_gain_stats(merged: pd.DataFrame, near_zero_threshold: float = NEAR_ZERO_GAIN_THRESHOLD) -> dict:
    """
    merged: output of per_plant_averaged_across_seeds(dfs), i.e. one row per
    plant with zero_shot_MAE_mean / adapted_MAE_mean already averaged across seeds.

    Returns gain statistics needed for Table "gain_stats":
      gain_abs = zero_shot_MAE_mean - adapted_MAE_mean   (positive = improvement)
      gain_pct = gain_abs / zero_shot_MAE_mean * 100
      mean_median_diff = mean_gain_abs - median_gain_abs
        (positive -> distribution is right-skewed, i.e. a few plants with
         very large gains pull the mean above the typical/median plant;
         negative -> left-skewed, mean pulled down by a few large negative
         outliers)
    """
    gain_abs = merged["zero_shot_MAE_mean"] - merged["adapted_MAE_mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_pct = np.where(
            merged["zero_shot_MAE_mean"] != 0,
            gain_abs / merged["zero_shot_MAE_mean"] * 100,
            np.nan,
        )

    n = len(gain_abs)
    mean_gain_abs = gain_abs.mean()
    median_gain_abs = gain_abs.median()
    pct_near_zero = 100 * (gain_abs.abs() < near_zero_threshold).sum() / n
    pct_negative = 100 * (gain_abs < -near_zero_threshold).sum() / n

    return {
        "mean_gain_abs": mean_gain_abs,
        "median_gain_abs": median_gain_abs,
        "mean_median_diff": mean_gain_abs - median_gain_abs,
        "std_gain_abs": gain_abs.std(ddof=1),
        "mean_gain_pct": np.nanmean(gain_pct),
        "pct_near_zero": pct_near_zero,
        "pct_negative": pct_negative,
    }

def to_latex_aggregate_performance(perf_df: pd.DataFrame) -> str:
    lines = []
    for _, row in perf_df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['ZeroShot_MAE_mean']:.4f}$\\pm${row['ZeroShot_MAE_std']:.4f} & "
            f"{row['Adapted_MAE_mean']:.4f}$\\pm${row['Adapted_MAE_std']:.4f} & "
            f"{row['Adapted_RMSE_mean']:.4f}$\\pm${row['Adapted_RMSE_std']:.4f} \\\\"
        )
    return "\n".join(lines)

def to_latex_gain_stats(gain_df: pd.DataFrame) -> str:
    lines = []
    for _, row in gain_df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['mean_gain_abs']:.4f} & {row['median_gain_abs']:.4f} & "
            f"{row['mean_median_diff']:.4f} & "
            f"{row['std_gain_abs']:.4f} & {row['mean_gain_pct']:.2f} & "
            f"{row['pct_near_zero']:.1f} & {row['pct_negative']:.1f} \\\\"
        )
    return "\n".join(lines)


def to_latex_speed_distribution(speed_dist_df: pd.DataFrame) -> str:
    lines = []
    for _, row in speed_dist_df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['mean']:.2f} & {row['median']:.2f} & "
            f"{row['min']:.2f} & {row['max']:.2f} & {row['std']:.2f} \\\\"
        )
    return "\n".join(lines)


def to_latex_speed_overall(speed_overall_df: pd.DataFrame, threshold_col: str) -> str:
    lines = []
    for _, row in speed_overall_df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row[f'{threshold_col}_mean']:.2f}$\\pm${row[f'{threshold_col}_std']:.2f} \\\\"
        )
    return "\n".join(lines)


def plot_gain_violin(gain_distributions: dict, out_path: Path):
    """
    gain_distributions: dict config_id -> {"gain_abs": np.array, "abbrev": str}

    Violin plot of per-plant gain_abs distribution per configuration, with
    the MEAN (red triangle, dashed line) and MEDIAN (black line) explicitly
    marked, so the mean-median gap is directly visible per config.
    """
    config_ids = list(gain_distributions.keys())
    n = len(config_ids)
    if n == 0:
        return

    data = [gain_distributions[c]["gain_abs"] for c in config_ids]
    labels = [gain_distributions[c]["abbrev"] for c in config_ids]

    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 5.5))

    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)

    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.4)

    # style mean vs. median lines distinctly
    parts["cmeans"].set_color("tab:red")
    parts["cmeans"].set_linewidth(2)
    parts["cmeans"].set_linestyle("--")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"Absolute adaptation gain ($\mathrm{gain\_abs}$)", fontsize=14)
    ax.set_title("Per-plant adaptation gain distribution (mean vs. median)", fontsize=16)
    ax.grid(alpha=0.3, axis="y")
    

    # custom legend (violinplot parts aren't directly legend-friendly)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="tab:red", linestyle="--", linewidth=2, label="Mean"),
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
    ]
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(handles=legend_elems, loc="upper right", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════
# PART D — Sample efficiency aggregation (MAE / masked loss vs. k-shot)
# ═══════════════════════════════════════════════════════════════════

_sample_eff_job_dir_cache: dict = {}


def find_job_dir_for_sample_efficiency(job_id: str) -> Path:
    """
    Locate output/<name_of_the_folder>/plots/job_<job_id>/.

    Only "<name_of_the_folder>" is unknown/variable, so only THAT single
    level is scanned (one `iterdir()` over `output/`); everything below it
    is a fixed-depth check (`.../plots/job_<id>/`). This deliberately avoids
    any recursive glob/rglob/os.walk over the full output tree, which would
    be infeasible given how many files exist deeper down. Result is cached
    per job_id since both sample-efficiency files live in the same folder.
    """
    if job_id in _sample_eff_job_dir_cache:
        return _sample_eff_job_dir_cache[job_id]

    target_name = f"job_{job_id}"
    matches = []
    for folder in SAMPLE_EFF_SEARCH_ROOT.iterdir():
        if not folder.is_dir():
            continue
        candidate = folder / SAMPLE_EFF_SUBDIR / target_name
        if candidate.is_dir():
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"No sample-efficiency directory found for job_id={job_id} "
            f"(looked for {SAMPLE_EFF_SEARCH_ROOT}/*/{SAMPLE_EFF_SUBDIR}/{target_name}/)"
        )
    if len(matches) > 1:
        print(f"  [warn] multiple sample-efficiency dirs match job_id={job_id}: "
              f"{[str(m) for m in matches]} -> using {matches[0]}")

    _sample_eff_job_dir_cache[job_id] = matches[0]
    return matches[0]


def load_sample_efficiency_csv(job_id: str, metric_key: str) -> pd.DataFrame:
    job_dir = find_job_dir_for_sample_efficiency(job_id)
    path = job_dir / SAMPLE_EFF_FILES[metric_key]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {"k_shot", "hours", "n", "mean", "median", "min", "max", "q10", "q90"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df.sort_values("k_shot").reset_index(drop=True)

def load_adapted_rmse(job_id: str) -> float:
    """
    Liest den (ADAPTED) RMSE aus output/<>/plots/job_<job_id>/aggregate_metrics.txt.
    Die Datei wird NUR nach der Adaptation geschrieben -> der dort stehende
    RMSE-Wert ist automatisch der adapted RMSE (keine Zero-Shot-Trennung nötig).
    """
    job_dir = find_job_dir_for_sample_efficiency(job_id)  # Pfad-Logik wiederverwendet
    path = job_dir / "aggregate_metrics.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"RMSE\s*:\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not find 'RMSE :' in {path}")
    return float(match.group(1))


def aggregate_adapted_rmse_across_seeds(job_ids: list) -> dict:
    """
    Mittelt den adapted RMSE über alle Seeds einer Config
    (analog zu seed_level_overall_mean für MAE in Part A).
    """
    vals = []
    for jid in job_ids:
        try:
            vals.append(load_adapted_rmse(jid))
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip adapted RMSE, seed {jid}] {e}")
    vals = np.array(vals)
    return {
        "Adapted_RMSE_mean": vals.mean() if len(vals) else np.nan,
        "Adapted_RMSE_std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
    }


def aggregate_sample_efficiency_across_seeds(job_ids: list, metric_key: str) -> pd.DataFrame:
    """
    Combine per-seed sample-efficiency stats (each already averaged over
    tasks/plants within that seed) into mean +/- std ACROSS SEEDS, per
    k_shot -- this is the quantity that should be reported to avoid seed
    noise. `hours` is taken from the first available seed (assumed
    identical support-window schedule across seeds of the same config).
    q10/q90 are additionally averaged across seeds for context only; the
    primary uncertainty measure to report is `std_across_seeds`.
    """
    seed_frames = []
    for jid in job_ids:
        try:
            df = load_sample_efficiency_csv(jid, metric_key)
        except FileNotFoundError as e:
            print(f"  [skip sample efficiency, seed {jid}, {metric_key}] {e}")
            continue
        seed_frames.append((jid, df))

    if not seed_frames:
        return pd.DataFrame()

    merged = seed_frames[0][1][["k_shot", "hours"]].drop_duplicates("k_shot")
    for jid, df in seed_frames:
        tmp = df[["k_shot", "mean"]].rename(columns={"mean": f"mean_seed{jid}"})
        merged = merged.merge(tmp, on="k_shot", how="inner")
    for jid, df in seed_frames:
        for col in ("q10", "q90"):
            tmp = df[["k_shot", col]].rename(columns={col: f"{col}_seed{jid}"})
            merged = merged.merge(tmp, on="k_shot", how="left")

    seed_cols = [c for c in merged.columns if c.startswith("mean_seed")]
    merged["mean_across_seeds"] = merged[seed_cols].mean(axis=1)
    merged["std_across_seeds"] = (
        merged[seed_cols].std(axis=1, ddof=1) if len(seed_cols) > 1 else np.nan
    )
    merged["q10_mean_across_seeds"] = merged[[c for c in merged.columns if c.startswith("q10_seed")]].mean(axis=1)
    merged["q90_mean_across_seeds"] = merged[[c for c in merged.columns if c.startswith("q90_seed")]].mean(axis=1)

    cols_keep = ["k_shot", "hours", "mean_across_seeds", "std_across_seeds",
                 "q10_mean_across_seeds", "q90_mean_across_seeds"] + seed_cols
    return merged[cols_keep].sort_values("k_shot").reset_index(drop=True)


def plot_sample_efficiency_curve(curve_df: pd.DataFrame, title: str, ylabel: str, out_path: Path):
    if curve_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_df["k_shot"], curve_df["mean_across_seeds"],
             marker="o", color="tab:green", label="mean (across seeds)")
    if curve_df["std_across_seeds"].notna().any():
        ax.fill_between(
            curve_df["k_shot"],
            curve_df["mean_across_seeds"] - curve_df["std_across_seeds"],
            curve_df["mean_across_seeds"] + curve_df["std_across_seeds"],
            alpha=0.25, color="tab:green", label="std (across seeds)",
        )
    ax.set_xlabel("Number of support windows (k-shot)")
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs='all', numticks=15))
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sample_efficiency_comparison(curves: dict, ylabel: str, out_path: Path):
    """curves: dict config_id -> {"df": curve_df, "abbrev": str}"""
    valid = {k: v for k, v in curves.items() if not v["df"].empty}
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (config_id, data) in enumerate(valid.items()):
        df = data["df"]
        color = colors[i % len(colors)]
        ax.plot(df["k_shot"], df["mean_across_seeds"], marker="o",
                 color=color, label=data["abbrev"], markersize=4)
        if df["std_across_seeds"].notna().any():
            ax.fill_between(
                df["k_shot"],
                df["mean_across_seeds"] - df["std_across_seeds"],
                df["mean_across_seeds"] + df["std_across_seeds"],
                alpha=0.12, color=color,
            )
    ax.set_xlabel("Number of support windows (k-shot)")
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs='all', numticks=15))
    ax.set_title(f"Sample efficiency: {ylabel} vs. support size")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


def sample_efficiency_summary_row(config_id: str, abbrev: str, curve_df: pd.DataFrame,
                                   checkpoints=SAMPLE_EFF_CHECKPOINTS) -> dict:
    row = {"Config": config_id, "Abbreviation": abbrev}
    indexed = curve_df.set_index("k_shot") if not curve_df.empty else None
    for k in checkpoints:
        if indexed is not None and k in indexed.index:
            row[f"k{k}_mean"] = indexed.loc[k, "mean_across_seeds"]
            row[f"k{k}_std"] = indexed.loc[k, "std_across_seeds"]
        else:
            row[f"k{k}_mean"] = np.nan
            row[f"k{k}_std"] = np.nan
    return row


def to_latex_sample_efficiency_summary(df: pd.DataFrame, checkpoints=SAMPLE_EFF_CHECKPOINTS) -> str:
    lines = []
    for _, row in df.iterrows():
        cells = [f"{row['Config']}", f"{row['Abbreviation']}"]
        for k in checkpoints:
            m, s = row.get(f"k{k}_mean"), row.get(f"k{k}_std")
            if pd.isna(m):
                cells.append("--")
            else:
                cells.append(f"{m:.4f}$\\pm${s:.4f}" if pd.notna(s) else f"{m:.4f}")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PART E — Final test-set distribution/calibration metrics
#          (ks_A_mean, ks_B_mean, coverage_mean) from SLURM .out logs
# ═══════════════════════════════════════════════════════════════════

_out_log_path_cache: Dict[str, Path] = {}


def find_out_log_file(job_id: str) -> Path:
    """
    Locate output/logs/<unknown_subfolder>/*_<job_id>.out.

    Only the single subfolder level under output/logs is unknown, so we
    iterate over `output/logs/*` ONCE (non-recursive `iterdir()`), and for
    each candidate subfolder do a cheap non-recursive glob for
    `*_<job_id>.out`. We also check directly inside `output/logs/` itself.
    This deliberately avoids a recursive rglob/os.walk over the whole logs
    tree. Result is cached per job_id.
    """
    if job_id in _out_log_path_cache:
        return _out_log_path_cache[job_id]

    if not OUT_LOG_ROOT.is_dir():
        raise FileNotFoundError(f"Log root not found: {OUT_LOG_ROOT}")

    pattern = OUT_LOG_FILE_GLOB_TMPL.format(job_id=job_id)
    matches = list(OUT_LOG_ROOT.glob(pattern))  # directly in output/logs/
    for sub in OUT_LOG_ROOT.iterdir():
        if sub.is_dir():
            matches.extend(sub.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No .out log found for job_id={job_id} "
            f"(looked for {OUT_LOG_ROOT}/{pattern} and {OUT_LOG_ROOT}/*/{pattern})"
        )
    if len(matches) > 1:
        matches = sorted(matches)
        print(f"  [warn] multiple .out logs match job_id={job_id}: "
              f"{[str(m) for m in matches]} -> using {matches[0]}")

    _out_log_path_cache[job_id] = matches[0]
    return matches[0]


def tail_read_text(
    path: Path,
    chunk_size: int = OUT_LOG_TAIL_CHUNK,
    max_bytes: int = OUT_LOG_TAIL_MAX_BYTES,
    required_substrings: Optional[List[str]] = None,
) -> str:
    """
    Read a (potentially huge) log file starting from the END, in
    geometrically growing windows, stopping as soon as all
    `required_substrings` are present in the decoded tail (or the file
    start / `max_bytes` cap is reached). Avoids loading the entire file
    into memory -- these .out files can be very large due to per-batch
    tqdm progress-bar output, while the metrics table we need is always
    the very last thing printed.
    """
    file_size = path.stat().st_size
    read_size = min(chunk_size, file_size)
    text = ""

    with open(path, "rb") as f:
        while True:
            read_size = min(read_size, file_size, max_bytes)
            if read_size < file_size:
                f.seek(-read_size, os.SEEK_END)
            else:
                f.seek(0, os.SEEK_SET)
            text = f.read().decode("utf-8", errors="ignore")

            done_found = required_substrings is None or all(
                s in text for s in required_substrings
            )
            done_capped = read_size >= file_size or read_size >= max_bytes
            if done_found or done_capped:
                break
            read_size *= 4

    return text


def parse_final_test_metrics(
    path: Path, metric_keys: Dict[str, str] = FINAL_METRIC_KEYS
) -> Dict[str, float]:
    """
    Extract the final values of `metric_keys` (short_name -> log key, e.g.
    "ks_A_mean" -> "test/ks_A_mean") from the LAST-printed Lightning
    "Test metric" table in a (possibly huge) SLURM .out log. Only the tail
    of the file is read (see `tail_read_text`).
    """
    required = list(metric_keys.values())
    text = tail_read_text(path, required_substrings=required)

    results = {}
    for short_name, log_key in metric_keys.items():
        # Lines look like: "│   test/ks_A_mean   │   0.18121644854545593   │"
        # Match the key, then grab the trailing float, robust to varying
        # box-drawing characters / column widths across runs.
        pattern = re.compile(
            rf"{re.escape(log_key)}\s*[│|]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        results[short_name] = float(matches[-1]) if matches else np.nan
        if not matches:
            print(f"  [warn] '{log_key}' not found in tail of {path.name}")

    return results


def aggregate_final_metrics_across_seeds(job_ids: List[str]) -> pd.DataFrame:
    """
    One row per seed (job_id) with ks_A_mean / ks_B_mean / coverage_mean.
    Each .out log contributes exactly one scalar per metric (the final
    value printed by Lightning for that run) -- analogous to the
    adaptation-speed overall summary (Part B) -- so mean/std across these
    per-seed scalars is the natural way to de-noise a single-run estimate.
    """
    rows = []
    for jid in job_ids:
        try:
            log_path = find_out_log_file(jid)
        except FileNotFoundError as e:
            print(f"  [skip final metrics, seed {jid}] {e}")
            continue
        metrics = parse_final_test_metrics(log_path)
        rows.append({"job_id": jid, **metrics})
    return pd.DataFrame(rows)


def summarize_final_metrics(per_seed_df: pd.DataFrame,
                             metric_keys=FINAL_METRIC_KEYS) -> dict:
    summary = {}
    for short_name in metric_keys:
        vals = per_seed_df.get(short_name, pd.Series(dtype=float)).dropna().to_numpy()
        summary[f"{short_name}_mean"] = vals.mean() if len(vals) else np.nan
        summary[f"{short_name}_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
    return summary


def to_latex_final_metrics(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['ks_A_mean_mean']:.4f}$\\pm${row['ks_A_mean_std']:.4f} & "
            f"{row['ks_B_mean_mean']:.4f}$\\pm${row['ks_B_mean_std']:.4f} & "
            f"{row['coverage_mean_mean']:.2f}$\\pm${row['coverage_mean_std']:.2f} \\\\"
        )
    return "\n".join(lines)


def plot_final_metrics_comparison(summary_rows: List[dict], out_path: Path):
    """
    Grouped bar chart: for each config, mean +/- std (across seeds) of
    ks_A_mean / ks_B_mean (left panel) and coverage_mean (right panel,
    different scale: %).
    """
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    labels = df["Abbreviation"].tolist()
    n = len(df)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.2 * n + 4), 4.5))

    axes[0].bar(x - width / 2, df["ks_A_mean_mean"], width,
                yerr=df["ks_A_mean_std"], label="Test A (Support vs Query)",
                color="tab:blue", capsize=3)
    axes[0].bar(x + width / 2, df["ks_B_mean_mean"], width,
                yerr=df["ks_B_mean_std"], label="Test B (Pred vs Query)",
                color="tab:orange", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("KS statistic D")
    axes[0].set_title("Covariate shift (A) vs. calibration (B)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, df["coverage_mean_mean"], width=0.5,
                yerr=df["coverage_mean_std"], color="tab:purple", capsize=3)
    axes[1].axhline(80.0, color="gray", linestyle="--", alpha=0.7, label="80% reference")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Coverage [%]")
    axes[1].set_title("Quantile coverage")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Final test-set distribution/calibration metrics (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════
# PART F — Best-/worst-case plant tables (Section 6.4)
# ═══════════════════════════════════════════════════════════════════

def build_full_plant_table(dfs, merged_base: pd.DataFrame) -> pd.DataFrame:
    """
    Extends per_plant_averaged_across_seeds() output with gain_abs/gain_pct
    (Eq. abs_gain/rel_gain) and, if present in the raw per-seed CSVs,
    rated_power_kW (averaged across seeds the same way as the MAE columns).
    """
    df = merged_base.copy()
    df["gain_abs"] = df["zero_shot_MAE_mean"] - df["adapted_MAE_mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["gain_pct"] = np.where(
            df["zero_shot_MAE_mean"] != 0,
            df["gain_abs"] / df["zero_shot_MAE_mean"] * 100,
            np.nan,
        )

    if "rated_power_kW" in dfs[0].columns:
        rp_cols = []
        for i, d in enumerate(dfs):
            tmp = d[["plant_id", "rated_power_kW"]].rename(
                columns={"rated_power_kW": f"rated_power_kW_seed{i}"}
            )
            df = df.merge(tmp, on="plant_id", how="left")
            rp_cols.append(f"rated_power_kW_seed{i}")
        df["rated_power_kW"] = df[rp_cols].mean(axis=1)
    else:
        df["rated_power_kW"] = np.nan
        print("  [warn] 'rated_power_kW' not found in per_plant_test_results.csv; "
              "best/worst-case tables will show '--' for Rated Power.")

    return df[["plant_id", "rated_power_kW", "zero_shot_MAE_mean",
               "adapted_MAE_mean", "gain_abs", "gain_pct"]]


def select_topN(df: pd.DataFrame, column: str, n: int, ascending: bool) -> pd.DataFrame:
    return df.sort_values(column, ascending=ascending).head(n).reset_index(drop=True)


def to_latex_plant_rows(df: pd.DataFrame, extra_col: str = None) -> str:
    """
    extra_col: 'gain_abs' or 'gain_pct' to append a 5th column; None for the
    easiest/hardest zero-shot tables (4 columns only).
    """
    lines = []
    for _, row in df.iterrows():
        rp = f"{row['rated_power_kW']:.1f}" if pd.notna(row["rated_power_kW"]) else "--"
        pid_str = short_plant_id(row["plant_id"])
        base = (f"{pid_str} & {rp} & "
                f"{row['zero_shot_MAE_mean']:.4f} & {row['adapted_MAE_mean']:.4f}")
        if extra_col == "gain_abs":
            base += f" & {row['gain_abs']:.4f} \\\\"
        elif extra_col == "gain_pct":
            base += f" & {row['gain_pct']:.2f} \\\\"
        else:
            base += " \\\\"
        lines.append(base)
    return "\n".join(lines)


# def run_best_worst_case_analysis(plant_tables: dict, perf_df: pd.DataFrame,
#                                   n: int = BEST_WORST_N) -> Optional[str]:
#     """
#     Section 6.4: selects the best-performing configuration (lowest mean
#     adapted MAE across the configs that were actually computed) and
#     exports/prints the easiest/hardest zero-shot, best-gain (abs/pct), and
#     worst-gain (negative) plant tables for that configuration. Returns the
#     chosen config_id (or None if no data is available at all).
#     """
#     if perf_df.empty:
#         print("[warn] Part F skipped: aggregate_performance table is empty.")
#         return None

#     best_config_id = perf_df.loc[perf_df["Adapted_MAE_mean"].idxmin(), "Config"]
#     print(f"\n[info] Best-/worst-case analysis (Section 6.4) uses the "
#           f"best-performing configuration by mean adapted MAE: {best_config_id}")

#     if best_config_id not in plant_tables:
#         print(f"[warn] No per-plant table available for {best_config_id}; skipping Part F.")
#         return None

#     pdf = plant_tables[best_config_id]
#     n_total = len(pdf)

#     easiest   = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=True)
#     hardest   = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=False)
#     best_abs  = select_topN(pdf, "gain_abs", n, ascending=False)
#     best_pct  = select_topN(pdf, "gain_pct", n, ascending=False)
#     worst_abs = select_topN(pdf, "gain_abs", n, ascending=True)

#     n_negative = int((pdf["gain_abs"] < 0).sum())
#     pct_negative = 100 * n_negative / n_total if n_total else np.nan

#     for name, table in [("easiest_zeroshot", easiest), ("hardest_zeroshot", hardest),
#                          ("best_gain_abs", best_abs), ("best_gain_pct", best_pct),
#                          ("negative_gain", worst_abs)]:
#         table.to_csv(BEST_WORST_DIR / f"{best_config_id}_{name}.csv", index=False)

#     print(f"\n################ LaTeX body: easiest_zeroshot ({best_config_id}) ################\n")
#     print(to_latex_plant_rows(easiest))
#     print(f"\n################ LaTeX body: hardest_zeroshot ({best_config_id}) ################\n")
#     print(to_latex_plant_rows(hardest))
#     print(f"\n################ LaTeX body: best_gain_abs ({best_config_id}) ################\n")
#     print(to_latex_plant_rows(best_abs, extra_col="gain_abs"))
#     print(f"\n################ LaTeX body: best_gain_pct ({best_config_id}) ################\n")
#     print(to_latex_plant_rows(best_pct, extra_col="gain_pct"))
#     print(f"\n################ LaTeX body: negative_gain ({best_config_id}) ################\n")
#     print(to_latex_plant_rows(worst_abs, extra_col="gain_abs"))
#     print(f"\n[info] Fill into Section~6.4.3 text: {n_negative} out of {n_total} plants "
#           f"({pct_negative:.1f}%) show negative gain for {best_config_id}.")

#     # ---- Candidates for Table case_selection (Section 6.5) ----
#     median_val = pdf["adapted_MAE_mean"].median()
#     dist = (pdf["adapted_MAE_mean"] - median_val).abs()
#     best_overall  = pdf.loc[pdf["adapted_MAE_mean"].idxmin()]
#     worst_overall = pdf.loc[pdf["adapted_MAE_mean"].idxmax()]
#     typical       = pdf.loc[dist.idxmin()]
#     print(f"\n[info] Case-study candidates (Table case_selection), config {best_config_id}:")
#     print(f"  Best overall     : plant_id={best_overall['plant_id']}  "
#           f"adapted_MAE={best_overall['adapted_MAE_mean']:.4f}")
#     print(f"  Worst overall    : plant_id={worst_overall['plant_id']}  "
#           f"adapted_MAE={worst_overall['adapted_MAE_mean']:.4f}")
#     print(f"  'Good' (typical) : plant_id={typical['plant_id']}  "
#           f"adapted_MAE={typical['adapted_MAE_mean']:.4f}")
#     print("  Over-/under-forecasted: NOT selectable from per_plant_test_results.csv "
#           "(only unsigned MAE available) -- requires a signed bias column "
#           "(e.g. mean(pred_median - y_query) per plant); see clarification needed "
#           "for the forecast-curve case studies (Section 6.5.2).")

#     return best_config_id

# def _export_best_worst_tables_for_config(config_id: str, plant_tables: dict, n: int) -> None:
#     """Exports/prints the easiest/hardest zero-shot, best-gain (abs/pct), and
#     worst-gain (negative) plant tables + case-study candidates for one config_id."""
#     if config_id not in plant_tables:
#         print(f"[warn] No per-plant table available for {config_id}; skipping.")
#         return

#     pdf = plant_tables[config_id]
#     n_total = len(pdf)

#     easiest   = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=True)
#     hardest   = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=False)
#     best_abs  = select_topN(pdf, "gain_abs", n, ascending=False)
#     best_pct  = select_topN(pdf, "gain_pct", n, ascending=False)
#     worst_abs = select_topN(pdf, "gain_abs", n, ascending=True)

#     n_negative = int((pdf["gain_abs"] < 0).sum())
#     pct_negative = 100 * n_negative / n_total if n_total else np.nan

#     for name, table in [("easiest_zeroshot", easiest), ("hardest_zeroshot", hardest),
#                          ("best_gain_abs", best_abs), ("best_gain_pct", best_pct),
#                          ("negative_gain", worst_abs)]:
#         table.to_csv(BEST_WORST_DIR / f"{config_id}_{name}.csv", index=False)

#     print(f"\n################ LaTeX body: easiest_zeroshot ({config_id}) ################\n")
#     print(to_latex_plant_rows(easiest))
#     print(f"\n################ LaTeX body: hardest_zeroshot ({config_id}) ################\n")
#     print(to_latex_plant_rows(hardest))
#     print(f"\n################ LaTeX body: best_gain_abs ({config_id}) ################\n")
#     print(to_latex_plant_rows(best_abs, extra_col="gain_abs"))
#     print(f"\n################ LaTeX body: best_gain_pct ({config_id}) ################\n")
#     print(to_latex_plant_rows(best_pct, extra_col="gain_pct"))
#     print(f"\n################ LaTeX body: negative_gain ({config_id}) ################\n")
#     print(to_latex_plant_rows(worst_abs, extra_col="gain_abs"))
#     print(f"\n[info] Fill into Section~6.4.3 text: {n_negative} out of {n_total} plants "
#           f"({pct_negative:.1f}%) show negative gain for {config_id}.")

def _export_best_worst_tables_for_config(config_id: str, plant_tables: dict, n: int) -> None:
    if config_id not in plant_tables:
        print(f"[warn] No per-plant table available for {config_id}; skipping.")
        return

    pdf = plant_tables[config_id]
    n_total = len(pdf)

    easiest     = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=True)
    hardest     = select_topN(pdf, "zero_shot_MAE_mean", n, ascending=False)
    best_abs    = select_topN(pdf, "gain_abs", n, ascending=False)
    best_pct    = select_topN(pdf, "gain_pct", n, ascending=False)
    worst_abs   = select_topN(pdf, "gain_abs", n, ascending=True)
    best_final  = select_topN(pdf, "adapted_MAE_mean", n, ascending=True)  

    n_negative = int((pdf["gain_abs"] < 0).sum())
    pct_negative = 100 * n_negative / n_total if n_total else np.nan

    for name, table in [("easiest_zeroshot", easiest), ("hardest_zeroshot", hardest),
                         ("best_gain_abs", best_abs), ("best_gain_pct", best_pct),
                         ("negative_gain", worst_abs),
                         ("best_final_mae", best_final)]:            
        table.to_csv(BEST_WORST_DIR / f"{config_id}_{name}.csv", index=False)

    print(f"\n################ LaTeX body: easiest_zeroshot ({config_id}) ################\n")
    print(to_latex_plant_rows(easiest))
    print(f"\n################ LaTeX body: hardest_zeroshot ({config_id}) ################\n")
    print(to_latex_plant_rows(hardest))
    print(f"\n################ LaTeX body: best_gain_abs ({config_id}) ################\n")
    print(to_latex_plant_rows(best_abs, extra_col="gain_abs"))
    print(f"\n################ LaTeX body: best_gain_pct ({config_id}) ################\n")
    print(to_latex_plant_rows(best_pct, extra_col="gain_pct"))
    print(f"\n################ LaTeX body: negative_gain ({config_id}) ################\n")
    print(to_latex_plant_rows(worst_abs, extra_col="gain_abs"))
    print(f"\n################ LaTeX body: best_final_mae ({config_id}) ################\n")   
    print(to_latex_plant_rows(best_final))                                                     
    print(f"\n[info] Fill into Section~6.4.3 text: {n_negative} out of {n_total} plants "
          f"({pct_negative:.1f}%) show negative gain for {config_id}.")

    median_val = pdf["adapted_MAE_mean"].median()
    dist = (pdf["adapted_MAE_mean"] - median_val).abs()
    best_overall  = pdf.loc[pdf["adapted_MAE_mean"].idxmin()]
    worst_overall = pdf.loc[pdf["adapted_MAE_mean"].idxmax()]
    typical       = pdf.loc[dist.idxmin()]
    print(f"\n[info] Case-study candidates (Table case_selection), config {config_id}:")
    print(f"  Best overall     : plant_id={best_overall['plant_id']}  "
          f"adapted_MAE={best_overall['adapted_MAE_mean']:.4f}")
    print(f"  Worst overall    : plant_id={worst_overall['plant_id']}  "
          f"adapted_MAE={worst_overall['adapted_MAE_mean']:.4f}")
    print(f"  'Good' (typical) : plant_id={typical['plant_id']}  "
          f"adapted_MAE={typical['adapted_MAE_mean']:.4f}")
    print("  Over-/under-forecasted: NOT selectable from per_plant_test_results.csv "
          "(only unsigned MAE available) -- requires a signed bias column "
          "(e.g. mean(pred_median - y_query) per plant); see clarification needed "
          "for the forecast-curve case studies (Section 6.5.2).")

def short_plant_id(pid, n: int = 3) -> str:
    """Formats a plant_id as '...234' (last n digits), consistent with the
    cherry-pick grid figures, to avoid printing full plant IDs in the thesis."""
    s = str(pid)
    # strip a trailing '.0' if plant_id was read as float
    if s.endswith(".0"):
        s = s[:-2]
    return f"...{s[-n:]}" if len(s) > n else s

def run_best_worst_case_analysis(plant_tables: dict, perf_df: pd.DataFrame,
                                  n: int = BEST_WORST_N,
                                  extra_config_ids: Optional[List[str]] = None
                                  ) -> Optional[str]:
    """
    Section 6.4: selects the best-performing configuration (lowest mean
    adapted MAE across the configs that were actually computed) and
    exports/prints its tables, PLUS the same tables for every config_id in
    `extra_config_ids` (e.g. ["C6"]), so both can be compared side by side.
    Returns the automatically chosen best_config_id (or None if no data at all).
    """
    if perf_df.empty:
        print("[warn] Part F skipped: aggregate_performance table is empty.")
        return None

    best_config_id = perf_df.loc[perf_df["Adapted_MAE_mean"].idxmin(), "Config"]
    print(f"\n[info] Best-/worst-case analysis (Section 6.4) uses the "
          f"best-performing configuration by mean adapted MAE: {best_config_id}")

    config_ids_to_run = [best_config_id] + [
        c for c in (extra_config_ids or []) if c != best_config_id
    ]

    for config_id in config_ids_to_run:
        _export_best_worst_tables_for_config(config_id, plant_tables, n)

    return best_config_id

def get_backbone(abbrev: str) -> str:
    """
    Extracts the backbone architecture ('LSTM' or 'TFT') from a config
    abbreviation such as 'LSTM-FOMAML' or 'TFT-ANIL'.
    """
    upper = abbrev.upper()
    if upper.startswith("LSTM"):
        return "LSTM"
    if upper.startswith("TFT"):
        return "TFT"
    return "unknown"


def select_second_case_study_config(perf_df: pd.DataFrame, best_config_id: str) -> Optional[str]:
    """
    Selects a second configuration for the case-study plots (Section 6.5),
    preferring the best-performing configuration from a DIFFERENT backbone
    architecture than `best_config_id`, so the two backbones (LSTM/TFT) can
    be fairly compared with their respective best configuration. Falls back
    to the overall second-best configuration (by mean adapted MAE) if no
    other backbone is available.
    """
    if perf_df.empty or len(perf_df) < 2:
        return None

    ranked = perf_df.sort_values("Adapted_MAE_mean", ascending=True).reset_index(drop=True)
    best_backbone = get_backbone(
        ranked.loc[ranked["Config"] == best_config_id, "Abbreviation"].iloc[0]
    )

    other_backbone = ranked[
        (ranked["Config"] != best_config_id)
        & (ranked["Abbreviation"].apply(get_backbone) != best_backbone)
    ]
    if not other_backbone.empty:
        row = other_backbone.iloc[0]
        print(f"[info] Second case-study configuration (other backbone, "
              f"'{get_backbone(row['Abbreviation'])}'): {row['Config']} ({row['Abbreviation']})")
        return row["Config"]

    remaining = ranked[ranked["Config"] != best_config_id]
    if remaining.empty:
        return None
    row = remaining.iloc[0]
    print(f"[warn] No configuration from a different backbone found; falling back to "
          f"overall second-best configuration: {row['Config']} ({row['Abbreviation']})")
    return row["Config"]


# ═══════════════════════════════════════════════════════════════════
# PART H — Cross-configuration paired comparison (Section 6.7)
# ═══════════════════════════════════════════════════════════════════

def paired_config_comparison(config_a: str, config_b: str, merged_tables: dict,
                              configs_meta: dict, metric_col: str = "adapted_MAE_mean"
                              ) -> Optional[dict]:
    """
    Paired comparison on the plants common to both configs (inner join on
    plant_id). Negative mean_diff => config A has lower (better) MAE.
    """
    if config_a not in merged_tables or config_b not in merged_tables:
        return None
    df_a = merged_tables[config_a][["plant_id", metric_col]].rename(columns={metric_col: "a"})
    df_b = merged_tables[config_b][["plant_id", metric_col]].rename(columns={metric_col: "b"})
    paired = df_a.merge(df_b, on="plant_id", how="inner")
    if paired.empty:
        return None

    diff = paired["a"] - paired["b"]
    try:
        stat, pval = wilcoxon(paired["a"], paired["b"])
    except ValueError:
        # e.g. all differences are exactly zero
        stat, pval = np.nan, np.nan

    return {
        "Config A": config_a, "Abbrev A": configs_meta[config_a]["abbrev"],
        "Config B": config_b, "Abbrev B": configs_meta[config_b]["abbrev"],
        "n_common_plants": len(paired),
        "mean_diff_A_minus_B": diff.mean(),
        "pct_A_better": 100 * (diff < 0).mean(),
        "wilcoxon_stat": stat,
        "p_value": pval,
        "significant_0.05": bool(pval < 0.05) if pd.notna(pval) else False,
    }


def run_cross_config_comparisons(merged_tables: dict, configs_meta: dict,
                                  pair_groups: dict = CROSS_CONFIG_PAIRS) -> Dict[str, pd.DataFrame]:
    results = {}
    for group_name, pairs in pair_groups.items():
        rows = []
        for a, b in pairs:
            res = paired_config_comparison(a, b, merged_tables, configs_meta)
            if res is not None:
                rows.append(res)
            else:
                print(f"  [skip cross-config, {group_name}] {a} vs {b}: no overlapping data")
        results[group_name] = pd.DataFrame(rows)
    return results


def to_latex_cross_config(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['Abbrev A']} vs.\\ {row['Abbrev B']} & "
            f"{row['n_common_plants']} & {row['mean_diff_A_minus_B']:.4f} & "
            f"{row['pct_A_better']:.1f} & {row['p_value']:.4g} & "
            f"{'Yes' if row['significant_0.05'] else 'No'} \\\\"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PART I — Lightning TensorBoard training-curve aggregation
#          (meta_train/val/test loss, zero-shot loss, masked/unmasked)
# ═══════════════════════════════════════════════════════════════════

def find_lightning_log_dir(job_id: str) -> Path:
    """Locate lightning_logs/version_<job_id>."""
    candidate = LIGHTNING_LOGS_ROOT / f"version_{job_id}"
    if not candidate.is_dir():
        raise FileNotFoundError(f"No lightning log dir found: {candidate}")
    return candidate


def load_lightning_curves_for_job(job_id: str, tag_map: dict = LIGHTNING_CURVE_TAGS) -> pd.DataFrame:
    """
    Reads epoch-level scalar curves from one lightning_logs/version_<job_id>
    run via tbparse. Returns long-format DataFrame: step, metric, value
    (restricted to tag_map's keys, renamed to tag_map's values).
    """
    log_dir = find_lightning_log_dir(job_id)
    reader = read_lightning_logs.SummaryReader(log_dir)
    scalars = reader.scalars
    if scalars.empty:
        return pd.DataFrame()

    rows = []
    for raw_tag, friendly_name in tag_map.items():
        sub = scalars[scalars["tag"] == raw_tag][["step", "value"]].copy()
        if sub.empty:
            continue
        sub["metric"] = friendly_name
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# def aggregate_lightning_curves_across_seeds(job_ids: list,
#                                              tag_map: dict = LIGHTNING_CURVE_TAGS) -> Dict[str, pd.DataFrame]:
#     """
#     Returns dict: friendly_metric_name -> DataFrame[step, mean_seed<i>...,
#     mean_across_seeds, std_across_seeds]. Same merge-on-step pattern as
#     aggregate_step_curve_across_seeds (Part B) / aggregate_sample_efficiency_
#     across_seeds (Part D) -- mean +/- std ACROSS SEEDS, per epoch.
#     """
#     per_seed = []
#     for jid in job_ids:
#         try:
#             df = load_lightning_curves_for_job(jid, tag_map)
#         except FileNotFoundError as e:
#             print(f"  [skip lightning curves, seed {jid}] {e}")
#             continue
#         if not df.empty:
#             per_seed.append((jid, df))

#     result = {}
#     # for metric in set(tag_map.values()):
#     #     merged = None
#     #     for jid, df in per_seed:
#     #         sub = df[df["metric"] == metric][["step", "value"]].rename(
#     #             columns={"value": f"mean_seed{jid}"}
#     #         )
#     #         if sub.empty:
#     #             continue
#     #         merged = sub if merged is None else merged.merge(sub, on="step", how="inner")
#     #     if merged is None or merged.empty:
#     #         continue
#     for metric in set(tag_map.values()):
#         merged = None
#         for jid, df in per_seed:
#             sub = df[df["metric"] == metric][["step", "value"]].rename(
#                 columns={"value": f"mean_seed{jid}"}
#             )
#             if sub.empty:
#                 print(f"    [debug] {metric}: seed {jid} has NO rows for this tag")
#                 continue
#             print(f"    [debug] {metric}: seed {jid} has {len(sub)} rows, "
#                     f"step range [{sub['step'].min()}, {sub['step'].max()}]")
#             merged = sub if merged is None else merged.merge(sub, on="step", how="inner")
#             if merged is not None:
#                 print(f"    [debug] {metric}: after merging seed {jid}, "
#                         f"{len(merged)} rows remain")
#         seed_cols = [c for c in merged.columns if c.startswith("mean_seed")]
#         merged["mean_across_seeds"] = merged[seed_cols].mean(axis=1)
#         merged["std_across_seeds"] = (
#             merged[seed_cols].std(axis=1, ddof=1) if len(seed_cols) > 1 else np.nan
#         )
#         result[metric] = merged.sort_values("step").reset_index(drop=True)
#     return result

def aggregate_lightning_curves_across_seeds(job_ids: list,
                                             tag_map: dict = LIGHTNING_CURVE_TAGS) -> Dict[str, pd.DataFrame]:
    per_seed = []
    for jid in job_ids:
        try:
            df = load_lightning_curves_for_job(jid, tag_map)
        except FileNotFoundError as e:
            print(f"  [skip lightning curves, seed {jid}] {e}")
            continue
        if not df.empty:
            per_seed.append((jid, df))

    result = {}
    for metric in set(tag_map.values()):
        merged = None
        for jid, df in per_seed:
            sub = df[df["metric"] == metric][["step", "value"]].rename(
                columns={"value": f"mean_seed{jid}"}
            )
            if sub.empty:
                print(f"    [debug] {metric}: seed {jid} has NO rows for this tag")
                continue
            print(f"    [debug] {metric}: seed {jid} has {len(sub)} rows, "
                  f"step range [{sub['step'].min()}, {sub['step'].max()}]")
            merged = sub if merged is None else merged.merge(sub, on="step", how="inner")

        # --- THE FIX: skip metrics with no data across all seeds ---
        if merged is None or merged.empty:
            print(f"    [debug] {metric}: no data across seeds -> skipping")
            continue

        seed_cols = [c for c in merged.columns if c.startswith("mean_seed")]
        merged["mean_across_seeds"] = merged[seed_cols].mean(axis=1)
        merged["std_across_seeds"] = (
            merged[seed_cols].std(axis=1, ddof=1) if len(seed_cols) > 1 else np.nan
        )
        result[metric] = merged.sort_values("step").reset_index(drop=True)
    return result

def aggregate_lightning_scalar_across_seeds(job_ids: list, metric: str,
                                             tag_map: dict = LIGHTNING_CURVE_TAGS) -> dict:
    """
    For single-point metrics (e.g. meta_test_loss_masked,
    zero_shot_test_loss_masked), logged exactly ONCE per seed at
    whatever global_step `trainer.test()` happened to run at.

    Unlike aggregate_lightning_curves_across_seeds(), this does NOT
    merge/join on `step` across seeds (which silently drops seeds whose
    test-time step differs, causing near-total data loss -- see bug
    report). Instead, each seed's scalar is taken independently (by
    construction there should be exactly one row per seed for this
    metric), and mean/std are computed directly across seeds.

    Returns: {"mean_across_seeds": float, "std_across_seeds": float,
              "n_seeds": int, "per_seed_values": {job_id: value}}
    """
    raw_tag = {v: k for k, v in tag_map.items()}.get(metric)
    if raw_tag is None:
        raise ValueError(f"Unknown metric '{metric}' (not found in tag_map values).")

    per_seed_values = {}
    for jid in job_ids:
        try:
            log_dir = find_lightning_log_dir(jid)
        except FileNotFoundError as e:
            print(f"  [skip scalar metric, seed {jid}] {e}")
            continue

        reader = read_lightning_logs.SummaryReader(log_dir)
        scalars = reader.scalars
        if scalars.empty:
            print(f"  [skip scalar metric, seed {jid}] no scalars found")
            continue

        sub = scalars[scalars["tag"] == raw_tag]
        if sub.empty:
            print(f"  [warn] metric '{metric}' (tag '{raw_tag}') not found for seed {jid}")
            continue
        if len(sub) > 1:
            print(f"  [warn] metric '{metric}' has {len(sub)} rows for seed {jid} "
                  f"(expected 1 -- test_step ran more than once?); using the LAST value.")

        per_seed_values[jid] = float(sub.iloc[-1]["value"])

    if not per_seed_values:
        return {"mean_across_seeds": np.nan, "std_across_seeds": np.nan,
                "n_seeds": 0, "per_seed_values": {}}

    vals = np.array(list(per_seed_values.values()))
    return {
        "mean_across_seeds": vals.mean(),
        "std_across_seeds": vals.std(ddof=1) if len(vals) > 1 else np.nan,
        "n_seeds": len(vals),
        "per_seed_values": per_seed_values,
    }
def plot_lightning_curve_single_config(curve_df: pd.DataFrame, metric_label: str,
                                        config_id: str, abbrev: str, out_path: Path):
    """
    Per-config figure: mean +/- std (across seeds) of one metric vs. epoch.
    Same style as plot_adaptation_curve_across_seeds / plot_sample_efficiency_curve.
    """
    if curve_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_df["step"], curve_df["mean_across_seeds"],
             marker="o", markersize=4, color="tab:blue", label="mean (across seeds)")
    if curve_df["std_across_seeds"].notna().any():
        ax.fill_between(
            curve_df["step"],
            curve_df["mean_across_seeds"] - curve_df["std_across_seeds"],
            curve_df["mean_across_seeds"] + curve_df["std_across_seeds"],
            alpha=0.2, color="tab:blue", label="std (across seeds)",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} — {config_id} ({abbrev})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lightning_metric_comparison(metric_curves: dict, metric_label: str, out_path: Path):
    """
    Comparison figure: mean +/- std (across seeds) of one metric vs. epoch,
    one line per configuration. metric_curves: dict config_id -> {"df":..., "abbrev":...}
    """
    valid = {k: v for k, v in metric_curves.items() if not v["df"].empty}
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (config_id, data) in enumerate(valid.items()):
        df = data["df"]
        color = colors[i % len(colors)]
        ax.plot(df["step"], df["mean_across_seeds"], marker="o", markersize=3,
                 color=color, label=data["abbrev"])
        if df["std_across_seeds"].notna().any():
            ax.fill_between(
                df["step"],
                df["mean_across_seeds"] - df["std_across_seeds"],
                df["mean_across_seeds"] + df["std_across_seeds"],
                alpha=0.12, color=color,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} across configurations")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


def lightning_summary_row(config_id: str, abbrev: str, metric: str, curve_df: pd.DataFrame) -> dict:
    """One row per (config, metric): final-epoch and best(min)-epoch mean+/-std across seeds."""
    if curve_df.empty:
        return {"Config": config_id, "Abbreviation": abbrev, "Metric": metric,
                "final_mean": np.nan, "final_std": np.nan,
                "best_mean": np.nan, "best_std": np.nan, "best_epoch": np.nan}
    final_row = curve_df.iloc[-1]
    best_idx = curve_df["mean_across_seeds"].idxmin()
    best_row = curve_df.loc[best_idx]
    return {
        "Config": config_id, "Abbreviation": abbrev, "Metric": metric,
        "final_mean": final_row["mean_across_seeds"], "final_std": final_row["std_across_seeds"],
        "best_mean": best_row["mean_across_seeds"], "best_std": best_row["std_across_seeds"],
        "best_epoch": best_row["step"],
    }


# ═══════════════════════════════════════════════════════════════════
# PART G — Case-study forecast curves (Section 6.5) from pre-generated
#          Plotly HTML (PredictionPlotEvaluator output)
# ═══════════════════════════════════════════════════════════════════

_pred_plots_job_dir_cache: Dict[str, Path] = {}


def find_job_dir_for_prediction_plots(job_id: str) -> Path:
    """
    Locate output/<unknown_folder>/prediction_plots/job_<job_id>/.
    Same one-level-scan pattern used for Parts D/E: only `output/*` is
    iterated once; everything below it is a fixed-depth check.
    """
    if job_id in _pred_plots_job_dir_cache:
        return _pred_plots_job_dir_cache[job_id]

    target_name = f"job_{job_id}"
    matches = []
    for folder in PRED_PLOTS_SEARCH_ROOT.iterdir():
        if not folder.is_dir():
            continue
        candidate = folder / PRED_PLOTS_SUBDIR / target_name
        if candidate.is_dir():
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"No prediction_plots directory found for job_id={job_id} "
            f"(looked for {PRED_PLOTS_SEARCH_ROOT}/*/{PRED_PLOTS_SUBDIR}/{target_name}/)"
        )
    if len(matches) > 1:
        print(f"  [warn] multiple prediction_plots dirs match job_id={job_id}: "
              f"{[str(m) for m in matches]} -> using {matches[0]}")

    _pred_plots_job_dir_cache[job_id] = matches[0]
    return matches[0]


def load_prediction_summary(job_dir: Path) -> pd.DataFrame:
    path = job_dir / PRED_PLOTS_SUMMARY_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {"group", "plant_id", "n_points", "MAE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def _skip_ws_and_comma(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    if pos < len(text) and text[pos] == ",":
        pos += 1
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
    return pos


def extract_plotly_figure_json(html_path: Path):
    """
    Parses the LAST `Plotly.newPlot("divid", DATA, LAYOUT, CONFIG)` call
    embedded in a Plotly `fig.write_html()` output.

    NOTE: the string "Plotly.newPlot(" also appears earlier in the file as
    part of the embedded plotly.js library's own minified source (e.g. in
    an error-message string documenting its own API), so a naive
    `text.find(...)` (first occurrence) can match library code instead of
    the actual render call. The real call with the figure's data/layout is
    always the LAST occurrence in the file, so `rfind` is used instead.
    """
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    marker = "Plotly.newPlot("
    start = text.rfind(marker)          # <-- changed from .find(...)
    if start == -1:
        raise ValueError(
            f"No 'Plotly.newPlot(' call found in {html_path.name} "
            f"(unexpected HTML structure / plotly.py version)."
        )
    pos = start + len(marker)
    decoder = json.JSONDecoder()

    pos = _skip_ws_and_comma(text, pos)
    _, pos = decoder.raw_decode(text, pos)       # div id (discarded)

    pos = _skip_ws_and_comma(text, pos)
    data, pos = decoder.raw_decode(text, pos)    # list of traces

    pos = _skip_ws_and_comma(text, pos)
    layout, pos = decoder.raw_decode(text, pos)  # layout dict

    return data, layout


def _decode_plotly_array(val) -> np.ndarray:
    """
    Plotly.py serializes numeric arrays either as plain JSON lists, or
    (newer versions) as a compact typed-array dict:
        {"dtype": "f8", "bdata": "<base64>"}
    Decodes either representation into a 1-D numpy array.
    """
    if val is None:
        return np.array([])
    if isinstance(val, dict) and "bdata" in val and "dtype" in val:
        raw = base64.b64decode(val["bdata"])
        dtype = _PLOTLY_DTYPE_MAP.get(val["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported plotly typed-array dtype: {val['dtype']}")
        return np.frombuffer(raw, dtype=dtype)
    return np.asarray(val)


def _find_trace(data: list, name_exact: str = None, name_regex: str = None):
    out = []
    for trace in data:
        name = str(trace.get("name", ""))
        if name_exact is not None and name == name_exact:
            out.append(trace)
        elif name_regex is not None and re.match(name_regex, name):
            out.append(trace)
    return out


def load_query_arrays(html_path: Path) -> dict:
    """From a `*_pred_vs_true.html` file, extract x, y_true, y_median, q_lo, q_hi."""
    data, _ = extract_plotly_figure_json(html_path)

    measured = _find_trace(data, name_exact="measured")
    median = _find_trace(data, name_regex=r"^q0?\.?5.*median.*$|.*median.*")
    qlines = _find_trace(data, name_regex=r"^q\d*\.?\d+$")  # plain "q0.10"/"q0.90", not "...median..."

    if not measured:
        raise ValueError(f"'measured' trace not found in {html_path.name}")
    if not median:
        raise ValueError(f"median trace not found in {html_path.name}")

    y_true = _decode_plotly_array(measured[0].get("y"))
    x = measured[0].get("x")
    y_med = _decode_plotly_array(median[0].get("y"))

    q_lo = q_hi = np.array([])
    if len(qlines) >= 2:
        arrs = [_decode_plotly_array(t.get("y")) for t in qlines[:2]]
        stacked = np.vstack(arrs)
        q_lo = stacked.min(axis=0)
        q_hi = stacked.max(axis=0)
    else:
        print(f"  [warn] quantile band traces not found/incomplete in {html_path.name}; "
              f"shaded band will be omitted.")

    return {"x": x, "y_true": y_true, "y_median": y_med, "q_lo": q_lo, "q_hi": q_hi}


def load_support_arrays(html_path: Path) -> dict:
    """From a `*_support.html` file, extract x, y_support."""
    data, _ = extract_plotly_figure_json(html_path)
    support = _find_trace(data, name_exact="support (measured)")
    if not support:
        raise ValueError(f"'support (measured)' trace not found in {html_path.name}")
    y_sup = _decode_plotly_array(support[0].get("y"))
    x_sup = support[0].get("x")
    return {"x": x_sup, "y_support": y_sup}


def load_case_study_plant(job_dir: Path, group: str, plant_id: str) -> dict:
    """
    Loads both HTML files (query + support) for one (group, plant_id) pair
    as written by PredictionPlotEvaluator, e.g.:
        best_plant_10126373736_pred_vs_true.html
        best_plant_10126373736_support.html
    """
    base = f"{group}_plant_{plant_id}"
    query = load_query_arrays(job_dir / f"{base}_pred_vs_true.html")
    support_path = job_dir / f"{base}_support.html"
    support = load_support_arrays(support_path) if support_path.exists() else {"x": None, "y_support": np.array([])}
    return {**query, **support, "plant_id": plant_id, "group": group}


def select_case_study_plants(summary_df: pd.DataFrame) -> Dict[str, str]:
    """
    Resolves the 5 case-study criteria (Section 6.5, Table case_selection)
    from the job's own pre-selected best/middle/worst cherries:
      - Best overall      : min-MAE plant within the "best" group
      - Worst overall      : max-MAE plant within the "worst" group
      - 'Good' (typical)  : median-MAE plant within the "middle" group
    Over-/under-forecasted plants are NOT distinguishable from MAE alone
    (unsigned); they are resolved here from the signed bias
    mean(y_median - y_true), computed over the SAME pool of plants already
    available in this job's cherry-pick (not the full 1,205 test plants --
    see printed caveat).
    """
    picks = {}
    best_grp = summary_df[summary_df["group"] == "best"]
    worst_grp = summary_df[summary_df["group"] == "worst"]
    mid_grp = summary_df[summary_df["group"] == "middle"]

    if not best_grp.empty:
        picks["Best overall"] = str(best_grp.loc[best_grp["MAE"].idxmin(), "plant_id"])
    if not worst_grp.empty:
        picks["Worst overall"] = str(worst_grp.loc[worst_grp["MAE"].idxmax(), "plant_id"])
    if not mid_grp.empty:
        med = mid_grp["MAE"].median()
        idx = (mid_grp["MAE"] - med).abs().idxmin()
        picks["'Good' (typical)"] = str(mid_grp.loc[idx, "plant_id"])

    return picks


def add_over_under_forecasted_picks(picks: Dict[str, str], job_dir: Path,
                                     summary_df: pd.DataFrame) -> Dict[str, str]:
    """
    Computes signed bias mean(y_median - y_true) for every plant already
    available in `summary_df` (the job's pre-selected cherries, typically
    15 plants: 5 best/middle/worst each) and picks the most positive
    (over-forecasted) / most negative (under-forecasted) among them.
    CAVEAT: restricted to this pool, not the full 1,205-plant test set.
    """
    biases = {}
    for _, row in summary_df.iterrows():
        pid, group = str(row["plant_id"]), row["group"]
        try:
            q = load_query_arrays(job_dir / f"{group}_plant_{pid}_pred_vs_true.html")
        except (FileNotFoundError, ValueError) as e:
            print(f"  [warn] could not load bias for {group}/{pid}: {e}")
            continue
        biases[(group, pid)] = float(np.nanmean(q["y_median"] - q["y_true"]))

    if biases:
        over_key = max(biases, key=biases.get)
        under_key = min(biases, key=biases.get)
        picks["Over-forecasted"] = over_key[1]
        picks["Under-forecasted"] = under_key[1]
        print(f"  [info] Over-/under-forecasted picked among {len(biases)} "
              f"pre-selected cherries (NOT the full 1,205 test plants). "
              f"bias(over)={biases[over_key]:.4f}, bias(under)={biases[under_key]:.4f}")
        # patch group lookup for the two new picks
        picks["_group_map"] = picks.get("_group_map", {})
        picks["_group_map"]["Over-forecasted"] = over_key[0]
        picks["_group_map"]["Under-forecasted"] = under_key[0]
    return picks


def plot_case_study_grid(selected: Dict[str, dict], out_path: Path):
    """
    Grid figure: one row per case-study plant, two columns
    (support | query with shaded [q_lo, q_hi] band + median + ground truth).
    Feeds Figure case_curves (Section 6.5.2).
    """
    criteria = list(selected.keys())
    n = len(criteria)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)

    for i, crit in enumerate(criteria):
        d = selected[crit]
        ax_sup, ax_qry = axes[i, 0], axes[i, 1]

        x_sup = d.get("x") if d.get("y_support", np.array([])).size and d.get("x_support") is None else d.get("x_support")
        y_sup = d.get("y_support", np.array([]))
        if y_sup.size:
            x_sup_plot = np.arange(len(y_sup)) if d.get("x_support") is None else d["x_support"]
            ax_sup.plot(x_sup_plot, y_sup, color="darkorange", linewidth=1, label="support (measured)")
        ax_sup.set_title(f"{crit}: Plant {d['plant_id']} ({d['group']}) — support", fontsize=9)
        ax_sup.grid(alpha=0.3)
        ax_sup.legend(fontsize=7)

        x_q = d["x"] if d["x"] is not None else np.arange(len(d["y_true"]))
        if d["q_lo"].size and d["q_hi"].size:
            ax_qry.fill_between(x_q, d["q_lo"], d["q_hi"], alpha=0.2, color="royalblue",
                                 label=r"$[q_{0.1}, q_{0.9}]$")
        ax_qry.plot(x_q, d["y_median"], color="royalblue", linewidth=1.3, label="median prediction")
        ax_qry.plot(x_q, d["y_true"], color="black", linewidth=1, label="measured")
        ax_qry.set_title(f"{crit}: Plant {d['plant_id']} — query", fontsize=9)
        ax_qry.grid(alpha=0.3)
        ax_qry.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


def plot_case_study_distributions(selected: Dict[str, dict], out_path_calib: Path,
                                   out_path_shift: Path, threshold: float = 0.0):
    """
    Reuses the Test A / Test B CDF-comparison style (sec:diagnostic_plots):
      Test A (covariate shift)  : support vs. query distribution per plant
      Test B (calibration)      : predicted median vs. query distribution per plant
    Feeds Figures case_covariate_shift and case_calibration (Section 6.5.3-6.5.4).
    """
    criteria = list(selected.keys())
    n = len(criteria)
    if n == 0:
        return

    fig_a, axes_a = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    fig_b, axes_b = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, crit in enumerate(criteria):
        d = selected[crit]
        y_true = d["y_true"][d["y_true"] > threshold]
        y_sup = d["y_support"][d["y_support"] > threshold] if d["y_support"].size else np.array([])
        y_med = d["y_median"][d["y_median"] > threshold]

        ax = axes_a[0, i]
        for arr, lbl, color in [(y_sup, "support", "#2196F3"), (y_true, "query", "#FF5722")]:
            if arr.size:
                sv = np.sort(arr)
                p = np.arange(1, len(sv) + 1) / len(sv)
                ax.plot(sv, p, label=f"{lbl} (n={len(arr)})", color=color, linewidth=1.5)
        ax.set_title(f"{crit}\nPlant {d['plant_id']} — Test A", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = axes_b[0, i]
        for arr, lbl, color in [(y_med, "pred. median", "#4CAF50"), (y_true, "query", "#FF5722")]:
            if arr.size:
                sv = np.sort(arr)
                p = np.arange(1, len(sv) + 1) / len(sv)
                ax2.plot(sv, p, label=f"{lbl} (n={len(arr)})", color=color, linewidth=1.5)
        ax2.set_title(f"{crit}\nPlant {d['plant_id']} — Test B", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

    fig_a.suptitle("Test A — Covariate shift (support vs. query), selected plants")
    fig_a.tight_layout()
    fig_a.savefig(out_path_shift, dpi=200)
    plt.close(fig_a)
    print(f"  [figure] saved -> {out_path_shift}")

    fig_b.suptitle("Test B — Calibration (predicted median vs. query), selected plants")
    fig_b.tight_layout()
    fig_b.savefig(out_path_calib, dpi=200)
    plt.close(fig_b)
    print(f"  [figure] saved -> {out_path_calib}")


def compute_case_coverage(selected: Dict[str, dict], threshold: float = 0.0) -> pd.DataFrame:
    """Table case_coverage (Section 6.5.3): % of masked query targets inside [q_lo, q_hi]."""
    rows = []
    for crit, d in selected.items():
        mask = d["y_true"] > threshold
        y, lo, hi = d["y_true"][mask], (d["q_lo"][mask] if d["q_lo"].size else None), \
                    (d["q_hi"][mask] if d["q_hi"].size else None)
        if lo is None or hi is None or len(y) == 0:
            cov = np.nan
        else:
            cov = 100.0 * np.mean((y >= lo) & (y <= hi))
        rows.append({"Criterion": crit, "plant_id": d["plant_id"], "Coverage_pct": cov})
    return pd.DataFrame(rows)


def build_case_selection_table(picks: Dict[str, str], plant_tables: dict, config_id: str) -> pd.DataFrame:
    """Table case_selection (Section 6.5.1): adds zero_shot/adapted MAE and gain from Part F's plant_tables."""
    pdf = plant_tables.get(config_id)
    rows = []
    for crit, pid in picks.items():
        if crit == "_group_map":
            continue
        row = {"Criterion": crit, "plant_id": pid}
        if pdf is not None:
            match = pdf[pdf["plant_id"].astype(str) == str(pid)]
            if not match.empty:
                row["zero_shot_MAE"] = match.iloc[0]["zero_shot_MAE_mean"]
                row["adapted_MAE"] = match.iloc[0]["adapted_MAE_mean"]
                row["gain_abs"] = match.iloc[0]["gain_abs"]
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex_case_selection(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        zs = f"{row['zero_shot_MAE']:.4f}" if "zero_shot_MAE" in row and pd.notna(row.get("zero_shot_MAE")) else "--"
        ad = f"{row['adapted_MAE']:.4f}" if "adapted_MAE" in row and pd.notna(row.get("adapted_MAE")) else "--"
        ga = f"{row['gain_abs']:.4f}" if "gain_abs" in row and pd.notna(row.get("gain_abs")) else "--"
        pid_str = short_plant_id(row["plant_id"])
        lines.append(f"{row['Criterion']} & {pid_str} & {zs} & {ad} & {ga} \\\\")
    return "\n".join(lines)


def to_latex_case_coverage(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        cov = f"{row['Coverage_pct']:.2f}" if pd.notna(row["Coverage_pct"]) else "--"
        pid_str = short_plant_id(row["plant_id"]) 
        lines.append(f"{pid_str} & {cov} \\\\")
    return "\n".join(lines)


def run_case_study_analysis(config_id: str, job_ids: List[str], plant_tables: dict,
                             figures_dir: Path):
    """
    Section 6.5 driver: uses the FIRST seed/job_id of `config_id` (prediction
    HTML plots are per-job, not merged across seeds -- unlike every other
    aggregate in this script). Resolves the 5 case-study plants, builds the
    forecast-curve grid figure, the Test A/B CDF figures, and the
    case_selection / case_coverage tables.
    """
    job_id = job_ids[0]
    print(f"\n[info] Case studies (Section 6.5) use config {config_id}, "
          f"job_id={job_id} (first seed only -- prediction plots are per-job).")
    print(f"[warn] Assuming this prediction pass reflects the ADAPTED model "
          f"(PredictionPlotEvaluator does not tag zero-shot vs. adapted in "
          f"its output). Confirm before using 'zero-shot and adapted' wording "
          f"in the Figure~case_curves caption.")

    try:
        job_dir = find_job_dir_for_prediction_plots(job_id)
        summary_df = load_prediction_summary(job_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip case studies] {e}")
        return

    picks = select_case_study_plants(summary_df)
    picks = add_over_under_forecasted_picks(picks, job_dir, summary_df)
    group_map = picks.pop("_group_map", {})

    selected = {}
    for crit, pid in picks.items():
        group = group_map.get(crit) or summary_df.loc[summary_df["plant_id"].astype(str) == pid, "group"].iloc[0]
        try:
            selected[crit] = load_case_study_plant(job_dir, group, pid)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [warn] could not load case-study plant {crit} ({pid}): {e}")

    if not selected:
        print("  [skip case studies] no plant data could be loaded.")
        return

    # plot_case_study_grid(selected, figures_dir / "case_curves.pdf")
    # plot_case_study_distributions(
    #     selected,
    #     out_path_calib=figures_dir / "case_calibration.pdf",
    #     out_path_shift=figures_dir / "case_covariate_shift.pdf",
    # )
    plot_case_study_grid(selected, figures_dir / f"case_curves_{config_id}.pdf")
    plot_case_study_distributions(
        selected,
        out_path_calib=figures_dir / f"case_calibration_{config_id}.pdf",
        out_path_shift=figures_dir / f"case_covariate_shift_{config_id}.pdf",
    )
    coverage_df = compute_case_coverage(selected)
    coverage_df.to_csv(CASE_STUDY_DIR / f"{config_id}_case_coverage.csv", index=False)
    print(f"\n################ Table: case_coverage ({config_id}) ################\n")
    print(coverage_df.to_string(index=False))
    print(f"\n################ LaTeX body: case_coverage ################\n")
    print(to_latex_case_coverage(coverage_df))

    selection_df = build_case_selection_table(picks, plant_tables, config_id)
    selection_df.to_csv(CASE_STUDY_DIR / f"{config_id}_case_selection.csv", index=False)
    print(f"\n################ Table: case_selection ({config_id}) ################\n")
    print(selection_df.to_string(index=False))
    print(f"\n################ LaTeX body: case_selection ################\n")
    print(to_latex_case_selection(selection_df))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    stats_rows = []          # -> Table "aggregate_stats"
    perf_rows = []           # -> Table "aggregate_performance" (MAE cols)
    gain_rows = []           # -> Table "gain_stats"
    speed_dist_rows = []     # -> steps-to-threshold distribution (across tasks, seeds concatenated)
    speed_overall_rows = []  # -> steps-to-threshold scalar (mean +/- std across seeds)
    auc_dist_rows = []       # -> AUC distribution across tasks (seeds pooled)
    auc_overall_rows = []    # -> AUC scalar, mean +/- std across seeds
    per_plant_data = {}      # for aggregate_box figure
    gain_distributions = {}  # for gain_violin figure
    sample_eff_curves = {m: {} for m in SAMPLE_EFF_FILES}
    sample_eff_summary_rows = {m: [] for m in SAMPLE_EFF_FILES}
    plant_tables = {}        # Part F: per-config, per-plant table (gain_abs/gain_pct included)
    merged_tables = {}       # Part H: per-config, per-plant MAE table (seed-averaged)
    final_metrics_rows = []  # -> Table "final_test_metrics_summary"
    lightning_curves_all: Dict[str, dict] = {}  # Part I: metric -> {config_id: {"df":..., "abbrev":...}}
    lightning_summary_rows = []                  # Part I: flat summary table

    for config_id, meta in CONFIGS.items():
        abbrev = meta["abbrev"]
        job_ids = meta["job_ids"]
        print(f"\n{'='*80}\n{config_id} ({abbrev}) : jobs {job_ids}\n{'='*80}")

        # ---------- PART A: MAE from per_plant_test_results.csv ----------
        try:
            dfs = load_config_frames(job_ids)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip MAE] {e}")
            dfs = None

        if dfs is not None:
            zs_seed_means = seed_level_overall_mean(dfs, "zero_shot_MAE")
            ad_seed_means = seed_level_overall_mean(dfs, "adapted_MAE")

            # perf_rows.append({
            #     "Config": config_id,
            #     "Abbreviation": abbrev,
            #     "ZeroShot_MAE_mean": zs_seed_means.mean(),
            #     "ZeroShot_MAE_std": zs_seed_means.std(ddof=1) if len(zs_seed_means) > 1 else np.nan,
            #     "Adapted_MAE_mean": ad_seed_means.mean(),
            #     "Adapted_MAE_std": ad_seed_means.std(ddof=1) if len(ad_seed_means) > 1 else np.nan,
            # })

            rmse_stats = aggregate_adapted_rmse_across_seeds(job_ids)
            print(f"  [adapted RMSE] {rmse_stats}")

            perf_rows.append({
                "Config": config_id,
                "Abbreviation": abbrev,
                "ZeroShot_MAE_mean": zs_seed_means.mean(),
                "ZeroShot_MAE_std": zs_seed_means.std(ddof=1) if len(zs_seed_means) > 1 else np.nan,
                "Adapted_MAE_mean": ad_seed_means.mean(),
                "Adapted_MAE_std": ad_seed_means.std(ddof=1) if len(ad_seed_means) > 1 else np.nan,
                **rmse_stats,   # fügt Adapted_RMSE_mean / Adapted_RMSE_std hinzu
            })
            merged = per_plant_averaged_across_seeds(dfs)
            plant_tables[config_id] = build_full_plant_table(dfs, merged)
            merged_tables[config_id] = merged   # used by Part H
            per_plant_data[config_id] = {
                "zero_shot": merged["zero_shot_MAE_mean"].to_numpy(),
                "adapted": merged["adapted_MAE_mean"].to_numpy(),
                "abbrev": abbrev,
            }
            zs_stats = distribution_stats(merged["zero_shot_MAE_mean"])
            ad_stats = distribution_stats(merged["adapted_MAE_mean"])

            stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Zero-Shot", **zs_stats})
            stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Adapted", **ad_stats})

            print(f"  [MAE] Zero-Shot : {zs_stats}")
            print(f"  [MAE] Adapted   : {ad_stats}")

            # ---------- PART C: Adaptation gain ----------
            gain_stats = compute_gain_stats(merged)
            gain_rows.append({"Config": config_id, "Abbreviation": abbrev, **gain_stats})
            print(f"  [gain] {gain_stats}")

            gain_abs_series = merged["zero_shot_MAE_mean"] - merged["adapted_MAE_mean"]
            gain_distributions[config_id] = {
                "gain_abs": gain_abs_series.to_numpy(),
                "abbrev": abbrev,
            }

        # ---------- PART B: Adaptation speed from .err logs (+ AUC) ----------
        if config_id in NO_INNER_LOOP_CONFIGS:
            print(f"  [speed] skipped for {config_id} ({abbrev}): "
                  f"No-Inner-Loop configuration performs no adaptation step "
                  f"by construction, so adaptation-speed metrics are undefined.")
        else:
            step_summaries, steps_thr_dfs, auc_dfs = [], [], []
            for jid in job_ids:
                t0 = time.time()
                try:
                    _, step_summary, steps_thr, auc_df = compute_adaptation_speed_for_job(jid)
                except FileNotFoundError as e:
                    print(f"  [skip speed, seed {jid}] {e}")
                    continue
                print(f"  [timing] seed {jid} parse+speed took {time.time()-t0:.1f}s")
                if step_summary.empty:
                    print(f"  [skip speed, seed {jid}] no multi-step inner-loop data found "
                        f"(expected for Baseline/No-Inner-Loop)")
                    continue
                step_summaries.append(step_summary)
                steps_thr_dfs.append(steps_thr)
                auc_dfs.append(auc_df)

            if step_summaries:
                curve_df = aggregate_step_curve_across_seeds(step_summaries)
                curve_df.to_csv(SPEED_DIR / f"{config_id}_adaptation_curve.csv", index=False)
                plot_adaptation_curve_across_seeds(
                    curve_df,
                    title=f"Adaptation speed — {config_id} ({abbrev})",
                    out_path=SPEED_DIR / f"{config_id}_adaptation_curve.png",
                )
                print(f"  [speed] adaptation curve saved -> "
                    f"{SPEED_DIR / f'{config_id}_adaptation_curve.csv'}")

                threshold_col = f"steps_to_{int(CONVERGENCE_THRESHOLD*100)}pct"
                combined_thr = combine_steps_to_threshold_across_seeds(steps_thr_dfs, threshold_col)
                if not combined_thr.empty:
                    thr_stats = distribution_stats(combined_thr)
                    speed_dist_rows.append({"Config": config_id, "Abbreviation": abbrev, **thr_stats})

                    per_seed_overall = np.array(
                        [d[threshold_col].dropna().mean() for d in steps_thr_dfs
                        if not d.empty and threshold_col in d.columns]
                    )
                    speed_overall_rows.append({
                        "Config": config_id,
                        "Abbreviation": abbrev,
                        f"{threshold_col}_mean": per_seed_overall.mean(),
                        f"{threshold_col}_std": per_seed_overall.std(ddof=1) if len(per_seed_overall) > 1 else np.nan,
                    })
                    print(f"  [speed] steps-to-{int(CONVERGENCE_THRESHOLD*100)}% convergence: {thr_stats}")

                # ---------- AUC (un-normalized adaptation speed) ----------
                if auc_dfs:
                    combined_auc = combine_auc_across_seeds(auc_dfs)
                    if not combined_auc.empty:
                        auc_stats = distribution_stats(combined_auc)
                        auc_dist_rows.append({"Config": config_id, "Abbreviation": abbrev, **auc_stats})

                        per_seed_auc_mean = np.array(
                            [d["auc_mean"].dropna().mean() for d in auc_dfs if not d.empty]
                        )
                        auc_overall_rows.append({
                            "Config": config_id, "Abbreviation": abbrev,
                            "auc_mean_mean": per_seed_auc_mean.mean(),
                            "auc_mean_std": per_seed_auc_mean.std(ddof=1) if len(per_seed_auc_mean) > 1 else np.nan,
                        })
                        print(f"  [AUC] un-normalized adaptation-speed AUC: {auc_stats}")

                del curve_df
            else:
                print(f"  [speed] no inner-loop adaptation-speed data available for {config_id} "
                    f"(expected for Baseline/No-Inner-Loop)")

            del step_summaries, steps_thr_dfs, auc_dfs

        # ---------- PART D: Sample efficiency (support-window ablation) ----------
        for metric_key in SAMPLE_EFF_FILES:
            curve_df = aggregate_sample_efficiency_across_seeds(job_ids, metric_key)
            if curve_df.empty:
                print(f"  [sample eff] no data for {config_id} / {metric_key}")
                continue
            csv_path = SAMPLE_EFF_DIR / f"{config_id}_sample_efficiency_{metric_key}.csv"
            curve_df.to_csv(csv_path, index=False)
            plot_sample_efficiency_curve(
                curve_df,
                title=f"Sample efficiency ({SAMPLE_EFF_LABELS[metric_key]}) — {config_id} ({abbrev})",
                ylabel=SAMPLE_EFF_LABELS[metric_key],
                out_path=SAMPLE_EFF_DIR / f"{config_id}_sample_efficiency_{metric_key}.png",
            )
            print(f"  [sample eff] {metric_key} curve saved -> {csv_path}")
            sample_eff_curves[metric_key][config_id] = {"df": curve_df, "abbrev": abbrev}
            sample_eff_summary_rows[metric_key].append(
                sample_efficiency_summary_row(config_id, abbrev, curve_df)
            )

        # ---------- PART E: Final test-set distribution/calibration metrics ----------
        per_seed_df = aggregate_final_metrics_across_seeds(job_ids)
        if not per_seed_df.empty:
            per_seed_df.to_csv(
                FINAL_METRICS_DIR / f"{config_id}_final_test_metrics_per_seed.csv", index=False
            )
            summary = summarize_final_metrics(per_seed_df)
            final_metrics_rows.append({"Config": config_id, "Abbreviation": abbrev, **summary})
            print(f"  [final metrics] {config_id}: {summary}")
        else:
            print(f"  [final metrics] no data for {config_id}")

        # ---------- PART I: Lightning TensorBoard training curves ----------
        # curves_by_metric = aggregate_lightning_curves_across_seeds(job_ids)
        curve_tag_map = {k: v for k, v in LIGHTNING_CURVE_TAGS.items() if v in LIGHTNING_CURVE_METRICS}
        curves_by_metric = aggregate_lightning_curves_across_seeds(job_ids, tag_map=curve_tag_map)

        for metric in LIGHTNING_SCALAR_METRICS:
            res = aggregate_lightning_scalar_across_seeds(job_ids, metric)
            if res["n_seeds"] > 0:
                curves_by_metric[metric] = pd.DataFrame([{
                    "step": 0, "mean_across_seeds": res["mean_across_seeds"],
                    "std_across_seeds": res["std_across_seeds"],
                }])
        if not curves_by_metric:
            print(f"  [lightning] no TensorBoard curves found for {config_id} "
                  f"(check lightning_logs/version_<job_id> for job_ids={job_ids})")
        for metric, df in curves_by_metric.items():
            csv_path = LIGHTNING_DIR / f"{config_id}_{metric}.csv"
            df.to_csv(csv_path, index=False)

            fig_path = LIGHTNING_FIGURES_PER_CONFIG_DIR / f"{config_id}_{metric}.png"
            plot_lightning_curve_single_config(
                df, metric_label=LIGHTNING_METRIC_LABELS.get(metric, metric),
                config_id=config_id, abbrev=abbrev, out_path=fig_path,
            )
            print(f"  [lightning] {metric} curve saved -> {csv_path}")

            lightning_curves_all.setdefault(metric, {})[config_id] = {"df": df, "abbrev": abbrev}
            lightning_summary_rows.append(lightning_summary_row(config_id, abbrev, metric, df))
                # ---------- PART J: Cherry-pick 3×3 grid (seed-averaged plant table) ----------
        if config_id in plant_tables:
            try:
                out_png = CHERRY_PICK_DIR / f"{config_id}_cherry_pick_plots.png"
                cherry_pick_plots.make_cherry_pick_grid(
                    plant_tables[config_id],
                    top_n=CHERRY_PICK_TOP_N,
                    out_path=out_png,
                    title=(f"Cherry-Pick Analysis — {config_id} ({abbrev})  "
                           f"[seed-averaged per-plant table]"),
                    source_label=(f"{config_id} ({abbrev}), "
                                  f"seed-averaged across {len(job_ids)} seed(s)"),
                    show=False,
                )
                print(f"  [cherry-pick] grid saved -> {out_png}")
            except Exception as e:
                print(f"  [cherry-pick] skipped for {config_id}: {e}")
        else:
            print(f"  [cherry-pick] skipped for {config_id}: no plant table available.")
        
    # ---------------- Assemble dataframes ----------------
    stats_df = pd.DataFrame(stats_rows)
    perf_df = pd.DataFrame(perf_rows)
    gain_df = pd.DataFrame(gain_rows)
    speed_dist_df = pd.DataFrame(speed_dist_rows)
    speed_overall_df = pd.DataFrame(speed_overall_rows)
    auc_dist_df = pd.DataFrame(auc_dist_rows)
    auc_overall_df = pd.DataFrame(auc_overall_rows)

    # ---------------- Figures: Parts A/C/D ----------------
    FIGURES_DIR = Path("/mnt/data/kes/data/jdomingo/personal-new/thesis/figures/results")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_aggregate_box(per_plant_data, FIGURES_DIR / "aggregate_box.pdf")
    plot_gain_violin(gain_distributions, FIGURES_DIR / "gain_violin.pdf")

    for metric_key, label in SAMPLE_EFF_LABELS.items():
        plot_sample_efficiency_comparison(
            sample_eff_curves[metric_key], ylabel=label,
            out_path=FIGURES_DIR / f"sample_efficiency_{metric_key}.pdf",
        )
        summary_df = pd.DataFrame(sample_eff_summary_rows[metric_key])
        summary_df.to_csv(OUT_DIR / f"sample_efficiency_summary_{metric_key}.csv", index=False)

        print(f"\n\n################ Table: sample_efficiency_summary ({metric_key}) ################\n")
        print(summary_df.to_string(index=False) if not summary_df.empty else "(no data)")

        print(f"\n\n################ LaTeX body: sample_efficiency_summary ({metric_key}) ################\n")
        print(to_latex_sample_efficiency_summary(summary_df) if not summary_df.empty else "(no data)")

    # ---------------- Part E: final metrics (run ONCE, not per sample-eff metric) ----------------
    plot_final_metrics_comparison(final_metrics_rows, FIGURES_DIR / "final_test_metrics_comparison.pdf")
    final_metrics_df = pd.DataFrame(final_metrics_rows)
    final_metrics_df.to_csv(OUT_DIR / "final_test_metrics_summary.csv", index=False)

    print("\n\n################ Table: final_test_metrics_summary (ks_A/ks_B/coverage, mean+/-std over seeds) ################\n")
    print(final_metrics_df.to_string(index=False) if not final_metrics_df.empty else "(no data)")

    print("\n\n################ LaTeX body: final_test_metrics_summary ################\n")
    print(to_latex_final_metrics(final_metrics_df) if not final_metrics_df.empty else "(no data)")

    # ---------------- Part I: Lightning curve comparison figures + summary table ----------------
    for metric, curves in lightning_curves_all.items():
        plot_lightning_metric_comparison(
            curves, metric_label=LIGHTNING_METRIC_LABELS.get(metric, metric),
            out_path=FIGURES_DIR / f"lightning_{metric}.pdf",
        )

    lightning_summary_df = pd.DataFrame(lightning_summary_rows)
    lightning_summary_df.to_csv(OUT_DIR / "lightning_curves_summary.csv", index=False)
    print("\n\n################ Table: lightning_curves_summary (final/best epoch, mean+/-std over seeds) ################\n")
    print(lightning_summary_df.to_string(index=False) if not lightning_summary_df.empty else "(no data)")

    #     # ---------------- Part I: Lightning curve comparison figures + summary table ----------------
    # for metric, curves in lightning_curves_all.items():
    #     plot_lightning_metric_comparison(
    #         curves, metric_label=LIGHTNING_METRIC_LABELS.get(metric, metric),
    #         out_path=FIGURES_DIR / f"lightning_{metric}.pdf",
    #     )

    # lightning_summary_df = pd.DataFrame(lightning_summary_rows)
    # lightning_summary_df.to_csv(OUT_DIR / "lightning_curves_summary.csv", index=False)
    # print("\n\n################ Table: lightning_curves_summary (final/best epoch, mean+/-std over seeds) ################\n")
    # print(lightning_summary_df.to_string(index=False) if not lightning_summary_df.empty else "(no data)")

    # ---------- Final (single-point) test-loss summary, all configs ----------
    final_loss_rows = []
    for config_id, meta in CONFIGS.items():
        row = {"Config": config_id, "Abbreviation": meta["abbrev"]}
        for metric in ["meta_train_loss_masked", "meta_val_loss_masked",
                       "meta_test_loss_masked", "zero_shot_val_loss_masked",
                       "zero_shot_test_loss_masked"]:
            if metric in ("meta_test_loss_masked", "zero_shot_test_loss_masked"):
                res = aggregate_lightning_scalar_across_seeds(meta["job_ids"], metric)
                row[f"{metric}_mean"] = res["mean_across_seeds"]
                row[f"{metric}_std"] = res["std_across_seeds"]
                row[f"{metric}_n_seeds"] = res["n_seeds"]
            else:
                curve = lightning_curves_all.get(metric, {}).get(config_id, {}).get("df")
                if curve is not None and not curve.empty:
                    last = curve.iloc[-1]
                    row[f"{metric}_mean"] = last["mean_across_seeds"]
                    row[f"{metric}_std"] = last["std_across_seeds"]
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
        final_loss_rows.append(row)

    final_loss_df = pd.DataFrame(final_loss_rows)
    final_loss_df.to_csv(OUT_DIR / "final_losses_masked_summary.csv", index=False)
    print("\n\n################ Table: final_losses_masked_summary ################\n")
    print(final_loss_df.to_string(index=False))
    
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n\n################ Table: aggregate_stats (per-plant MAE distribution) ################\n")
    print(stats_df.to_string(index=False))

    print("\n\n################ Table: aggregate_performance (MAE cols, mean +/- std over seeds) ################\n")
    print(perf_df.to_string(index=False))

    print("\n\n################ LaTeX body: aggregate_performance ################\n")
    print(to_latex_aggregate_performance(perf_df) if not perf_df.empty else "(no data)")

    print("\n\n################ Table: gain_stats (adaptation gain by configuration) ################\n")
    print(gain_df.to_string(index=False) if not gain_df.empty else "(no data)")

    print("\n\n################ Table: adaptation speed — distribution across tasks (seeds concatenated) ################\n")
    print(speed_dist_df.to_string(index=False) if not speed_dist_df.empty else "(no data)")

    print("\n\n################ Table: adaptation speed — overall scalar, mean +/- std across seeds ################\n")
    print(speed_overall_df.to_string(index=False) if not speed_overall_df.empty else "(no data)")

    print("\n\n################ Table: adaptation AUC — distribution across tasks (seeds pooled) ################\n")
    print(auc_dist_df.to_string(index=False) if not auc_dist_df.empty else "(no data)")

    print("\n\n################ Table: adaptation AUC — overall scalar, mean +/- std across seeds ################\n")
    print(auc_overall_df.to_string(index=False) if not auc_overall_df.empty else "(no data)")

    stats_df.to_csv(OUT_DIR / "aggregate_stats.csv", index=False)
    perf_df.to_csv(OUT_DIR / "aggregate_performance_mae.csv", index=False)
    gain_df.to_csv(OUT_DIR / "gain_stats.csv", index=False)
    speed_dist_df.to_csv(OUT_DIR / "adaptation_speed_distribution.csv", index=False)
    speed_overall_df.to_csv(OUT_DIR / "adaptation_speed_overall.csv", index=False)
    auc_dist_df.to_csv(OUT_DIR / "adaptation_auc_distribution.csv", index=False)
    auc_overall_df.to_csv(OUT_DIR / "adaptation_auc_overall.csv", index=False)

    print("\n\n################ LaTeX body: gain_stats ################\n")
    print(to_latex_gain_stats(gain_df) if not gain_df.empty else "(no data)")

    threshold_col = f"steps_to_{int(CONVERGENCE_THRESHOLD*100)}pct"
    print("\n\n################ LaTeX body: adaptation_speed_stats (per-task distribution) ################\n")
    print(to_latex_speed_distribution(speed_dist_df) if not speed_dist_df.empty else "(no data)")

    print("\n\n################ LaTeX body: adaptation_speed_overall (mean +/- std over seeds) ################\n")
    print(to_latex_speed_overall(speed_overall_df, threshold_col) if not speed_overall_df.empty else "(no data)")

    print("\n\n################ LaTeX body: adaptation_auc_distribution (per-task distribution) ################\n")
    print(to_latex_speed_distribution(auc_dist_df) if not auc_dist_df.empty else "(no data)")

    print("\n\n################ LaTeX body: adaptation_auc_overall (mean +/- std over seeds) ################\n")
    print(to_latex_speed_overall(auc_overall_df, "auc_mean") if not auc_overall_df.empty else "(no data)")

    # ---------------- Part F: best-/worst-case plant tables (Section 6.4) --------------f--
    # best_config_id = run_best_worst_case_analysis(plant_tables, perf_df, n=BEST_WORST_N)

    best_config_id = run_best_worst_case_analysis(
        plant_tables, perf_df, n=BEST_WORST_N, extra_config_ids=["C6"]
    )



    # ---------------- Part G: case-study forecast curves (Section 6.5) ----------------
    # if best_config_id is not None:
    #     run_case_study_analysis(
    #         config_id=best_config_id,
    #         job_ids=CONFIGS[best_config_id]["job_ids"],
    #         plant_tables=plant_tables,
    #         figures_dir=FIGURES_DIR,
    #     )
    # else:
    #     print("[warn] Part G skipped: no best_config_id resolved from Part F.")

        # ---------------- Part G: case-study forecast curves (Section 6.5) ----------------
    if best_config_id is not None:
        second_config_id = select_second_case_study_config(perf_df, best_config_id)
        case_study_config_ids = [best_config_id] + ([second_config_id] if second_config_id else [])

        print(f"\n[info] Case-study configurations selected for Section 6.5: "
              f"{case_study_config_ids}")

        for cfg_id in case_study_config_ids:
            run_case_study_analysis(
                config_id=cfg_id,
                job_ids=CONFIGS[cfg_id]["job_ids"],
                plant_tables=plant_tables,
                figures_dir=FIGURES_DIR,
            )
    else:
        print("[warn] Part G skipped: no best_config_id resolved from Part F.")

        # ---------------- Part H: cross-configuration comparison (Section 6.7) ----------------
    cross_results = run_cross_config_comparisons(merged_tables, CONFIGS)
    for group_name, df in cross_results.items():
        df.to_csv(CROSS_CONFIG_DIR / f"{group_name}.csv", index=False)
        print(f"\n\n################ Table: cross_config — {group_name} ################\n")
        print(df.to_string(index=False) if not df.empty else "(no data)")
        print(f"\n################ LaTeX body: cross_config — {group_name} ################\n")
        print(to_latex_cross_config(df) if not df.empty else "(no data)")

    print(f"\nSaved CSVs to: {OUT_DIR}/")
    print(f"Per-config adaptation curves/plots saved to: {SPEED_DIR}/")
    print(f"Best-/worst-case tables saved to: {BEST_WORST_DIR}/")
    print(f"Cross-configuration comparison tables saved to: {CROSS_CONFIG_DIR}/")
    print(f"Lightning training curves saved to: {LIGHTNING_DIR}/")
    print(f"Cherry-pick grids saved to: {CHERRY_PICK_DIR}/")


if __name__ == "__main__":
    main()