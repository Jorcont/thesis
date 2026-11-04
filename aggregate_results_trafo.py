# """
# aggregate_results_trafo.py

# Simplified aggregation script for TRAFO configs, single seed per config
# (no averaging across seeds). Produces:

#   (A) Zero-shot MAE and adapted MAE per plant -> summary table + CSV + LaTeX
#   (E) Final test-set distribution/calibration metrics (Test A, Test B,
#       Coverage) from the SLURM .out logs -> summary table + CSV + LaTeX

# Run:
#     python aggregate_results_trafo.py            # all configs in CONFIGS
#     python aggregate_results_trafo.py C1         # only C1
#     python aggregate_results_trafo.py C1 C4      # only C1 and C4
# """

# import os
# import re
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional

# import numpy as np
# import pandas as pd

# # ----------------------------------------------------------------------
# # 1) EDIT THIS: configurations, abbreviations, job IDs (ONE per config)
# # ----------------------------------------------------------------------
# BASE_DIR = Path("output/pv_timeseries_tft_v2")
# CSV_NAME = "per_plant_test_results.csv"

# CONFIGS = {
#     # "C1": {"abbrev": "LSTM-Trafo-Base",   "job_ids": ""},
#     # "C2": {"abbrev": "TFT-Trafo-Base",    "job_ids": ""},
#     # "C3": {"abbrev": "LSTM-Trafo-FOMAML", "job_ids": ""},
#     "C4": {"abbrev": "TFT-Trafo-FOMAML",  "job_id": "352433"},
#     "C5": {"abbrev": "LSTM-Trafo-ANIL",   "job_id": "352819"},
#     "C6": {"abbrev": "TFT-Trafo-ANIL",    "job_id": "352211"},
#     # "C7": {"abbrev": "LSTM-Trafo-NIL",    "job_ids": ""},
#     "C8": {"abbrev": "TFT-Trafo-NIL",     "job_id": "352434"},
# }

# # ----------------------------------------------------------------------
# # Optional CLI filter: restrict to a subset of configs, e.g.
# #   python aggregate_results_trafo.py C1
# #   python aggregate_results_trafo.py C1 C4
# # ----------------------------------------------------------------------
# if len(sys.argv) > 1:
#     requested = sys.argv[1:]
#     unknown = [c for c in requested if c not in CONFIGS]
#     if unknown:
#         raise ValueError(f"Unknown config id(s): {unknown}. Available: {list(CONFIGS.keys())}")
#     CONFIGS = {k: v for k, v in CONFIGS.items() if k in requested}
#     print(f"[info] Running only: {list(CONFIGS.keys())}")

# # ----------------------------------------------------------------------
# # Output paths — same base path as the original script, but a dedicated
# # subfolder for the trafo-only results.
# # ----------------------------------------------------------------------
# OUT_DIR = Path("/mnt/data/kes/data/jdomingo/personal-new/thesis/results_trafo")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# FIGURES_DIR = OUT_DIR / "figures"
# FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# # Final test-metric logs (.out files)
# OUT_LOG_ROOT = Path("output/logs")
# OUT_LOG_FILE_GLOB_TMPL = "*_{job_id}.out"
# OUT_LOG_TAIL_CHUNK = 65536
# OUT_LOG_TAIL_MAX_BYTES = 5_000_000

# FINAL_METRIC_KEYS = {
#     "ks_A_mean": "test/ks_A_mean",
#     "ks_B_mean": "test/ks_B_mean",
#     "coverage_mean": "test/coverage_mean",
# }


# # ═══════════════════════════════════════════════════════════════════
# # PART A — Per-plant MAE (single seed, no averaging)
# # ═══════════════════════════════════════════════════════════════════

# def load_job_csv(job_id: str) -> pd.DataFrame:
#     path = BASE_DIR / f"job_{job_id}" / CSV_NAME
#     if not path.exists():
#         raise FileNotFoundError(f"Missing file: {path}")
#     df = pd.read_csv(path)
#     required = {"plant_id", "zero_shot_MAE", "adapted_MAE"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"{path} is missing columns: {missing}")
#     return df


# def distribution_stats(series: pd.Series) -> dict:
#     return {
#         "mean": series.mean(),
#         "median": series.median(),
#         "min": series.min(),
#         "max": series.max(),
#         "std": series.std(ddof=1),
#     }


# def to_latex_mae_stats(df: pd.DataFrame) -> str:
#     lines = []
#     for _, row in df.iterrows():
#         lines.append(
#             f"{row['Config']} & {row['Abbreviation']} & {row['Phase']} & "
#             f"{row['mean']:.4f} & {row['median']:.4f} & {row['min']:.4f} & "
#             f"{row['max']:.4f} & {row['std']:.4f} \\\\"
#         )
#     return "\n".join(lines)


# def to_latex_mae_summary(df: pd.DataFrame) -> str:
#     lines = []
#     for _, row in df.iterrows():
#         lines.append(
#             f"{row['Config']} & {row['Abbreviation']} & "
#             f"{row['ZeroShot_MAE']:.4f} & {row['Adapted_MAE']:.4f} \\\\"
#         )
#     return "\n".join(lines)


# # ═══════════════════════════════════════════════════════════════════
# # PART E — Final test-set distribution/calibration metrics
# #          (ks_A_mean, ks_B_mean, coverage_mean) from SLURM .out logs
# # ═══════════════════════════════════════════════════════════════════

# _out_log_path_cache: Dict[str, Path] = {}


# def find_out_log_file(job_id: str) -> Path:
#     if job_id in _out_log_path_cache:
#         return _out_log_path_cache[job_id]

#     if not OUT_LOG_ROOT.is_dir():
#         raise FileNotFoundError(f"Log root not found: {OUT_LOG_ROOT}")

#     pattern = OUT_LOG_FILE_GLOB_TMPL.format(job_id=job_id)
#     matches = list(OUT_LOG_ROOT.glob(pattern))
#     for sub in OUT_LOG_ROOT.iterdir():
#         if sub.is_dir():
#             matches.extend(sub.glob(pattern))

#     if not matches:
#         raise FileNotFoundError(
#             f"No .out log found for job_id={job_id} "
#             f"(looked for {OUT_LOG_ROOT}/{pattern} and {OUT_LOG_ROOT}/*/{pattern})"
#         )
#     if len(matches) > 1:
#         matches = sorted(matches)
#         print(f"  [warn] multiple .out logs match job_id={job_id}: "
#               f"{[str(m) for m in matches]} -> using {matches[0]}")

#     _out_log_path_cache[job_id] = matches[0]
#     return matches[0]


# def tail_read_text(
#     path: Path,
#     chunk_size: int = OUT_LOG_TAIL_CHUNK,
#     max_bytes: int = OUT_LOG_TAIL_MAX_BYTES,
#     required_substrings: Optional[List[str]] = None,
# ) -> str:
#     file_size = path.stat().st_size
#     read_size = min(chunk_size, file_size)
#     text = ""

#     with open(path, "rb") as f:
#         while True:
#             read_size = min(read_size, file_size, max_bytes)
#             if read_size < file_size:
#                 f.seek(-read_size, os.SEEK_END)
#             else:
#                 f.seek(0, os.SEEK_SET)
#             text = f.read().decode("utf-8", errors="ignore")

#             done_found = required_substrings is None or all(
#                 s in text for s in required_substrings
#             )
#             done_capped = read_size >= file_size or read_size >= max_bytes
#             if done_found or done_capped:
#                 break
#             read_size *= 4

#     return text


# def parse_final_test_metrics(
#     path: Path, metric_keys: Dict[str, str] = FINAL_METRIC_KEYS
# ) -> Dict[str, float]:
#     required = list(metric_keys.values())
#     text = tail_read_text(path, required_substrings=required)

#     results = {}
#     for short_name, log_key in metric_keys.items():
#         pattern = re.compile(
#             rf"{re.escape(log_key)}\s*[│|]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
#         )
#         matches = pattern.findall(text)
#         results[short_name] = float(matches[-1]) if matches else np.nan
#         if not matches:
#             print(f"  [warn] '{log_key}' not found in tail of {path.name}")

#     return results


# def to_latex_final_metrics(df: pd.DataFrame) -> str:
#     lines = []
#     for _, row in df.iterrows():
#         lines.append(
#             f"{row['Config']} & {row['Abbreviation']} & "
#             f"{row['ks_A_mean']:.4f} & {row['ks_B_mean']:.4f} & "
#             f"{row['coverage_mean']:.2f} \\\\"
#         )
#     return "\n".join(lines)


# # ═══════════════════════════════════════════════════════════════════
# # MAIN
# # ═══════════════════════════════════════════════════════════════════

# def main():
#     stats_rows = []       # -> Table "aggregate_stats" (per-plant MAE distribution)
#     perf_rows = []        # -> Table "aggregate_performance" (single scalar per config)
#     final_metrics_rows = []  # -> Table "final_test_metrics"

#     for config_id, meta in CONFIGS.items():
#         abbrev = meta["abbrev"]
#         job_id = meta["job_id"]
#         print(f"\n{'='*80}\n{config_id} ({abbrev}) : job_id {job_id}\n{'='*80}")

#         # ---------- PART A: Zero-shot / Adapted MAE (single seed) ----------
#         try:
#             df = load_job_csv(job_id)
#         except (FileNotFoundError, ValueError) as e:
#             print(f"  [skip MAE] {e}")
#             df = None

#         if df is not None:
#             zs_stats = distribution_stats(df["zero_shot_MAE"])
#             ad_stats = distribution_stats(df["adapted_MAE"])

#             stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Zero-Shot", **zs_stats})
#             stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Adapted", **ad_stats})

#             perf_rows.append({
#                 "Config": config_id,
#                 "Abbreviation": abbrev,
#                 "ZeroShot_MAE": zs_stats["mean"],
#                 "Adapted_MAE": ad_stats["mean"],
#             })

#             print(f"  [MAE] Zero-Shot : {zs_stats}")
#             print(f"  [MAE] Adapted   : {ad_stats}")

#         # ---------- PART E: Final test-set distribution/calibration metrics ----------
#         try:
#             log_path = find_out_log_file(job_id)
#             metrics = parse_final_test_metrics(log_path)
#             final_metrics_rows.append({"Config": config_id, "Abbreviation": abbrev, **metrics})
#             print(f"  [final metrics] {metrics}")
#         except FileNotFoundError as e:
#             print(f"  [skip final metrics] {e}")

#     # ---------------- Assemble dataframes ----------------
#     stats_df = pd.DataFrame(stats_rows)
#     perf_df = pd.DataFrame(perf_rows)
#     final_metrics_df = pd.DataFrame(final_metrics_rows)

#     pd.set_option("display.float_format", lambda x: f"{x:.4f}")

#     print("\n\n################ Table: aggregate_stats (per-plant MAE distribution) ################\n")
#     print(stats_df.to_string(index=False) if not stats_df.empty else "(no data)")

#     print("\n\n################ Table: aggregate_performance (Zero-Shot vs. Adapted MAE) ################\n")
#     print(perf_df.to_string(index=False) if not perf_df.empty else "(no data)")

#     print("\n\n################ Table: final_test_metrics (Test A, Test B, Coverage) ################\n")
#     print(final_metrics_df.to_string(index=False) if not final_metrics_df.empty else "(no data)")

#     # ---------------- Save CSVs ----------------
#     stats_df.to_csv(OUT_DIR / "aggregate_stats.csv", index=False)
#     perf_df.to_csv(OUT_DIR / "aggregate_performance_mae.csv", index=False)
#     final_metrics_df.to_csv(OUT_DIR / "final_test_metrics.csv", index=False)

#     # ---------------- Save LaTeX bodies ----------------
#     latex_stats = to_latex_mae_stats(stats_df) if not stats_df.empty else "% no data"
#     latex_perf = to_latex_mae_summary(perf_df) if not perf_df.empty else "% no data"
#     latex_final = to_latex_final_metrics(final_metrics_df) if not final_metrics_df.empty else "% no data"

#     (OUT_DIR / "aggregate_stats.tex").write_text(latex_stats)
#     (OUT_DIR / "aggregate_performance_mae.tex").write_text(latex_perf)
#     (OUT_DIR / "final_test_metrics.tex").write_text(latex_final)

#     print("\n\n################ LaTeX body: aggregate_stats ################\n")
#     print(latex_stats)

#     print("\n\n################ LaTeX body: aggregate_performance ################\n")
#     print(latex_perf)

#     print("\n\n################ LaTeX body: final_test_metrics ################\n")
#     print(latex_final)

#     print(f"\nSaved CSVs/LaTeX tables to: {OUT_DIR}/")


# if __name__ == "__main__":
#     main()

"""
aggregate_results_trafo.py

Simplified aggregation script for TRAFO configs, single seed per config
(no averaging across seeds). Produces:

  (A) Zero-shot MAE and adapted MAE per plant -> summary table + CSV + LaTeX
  (E) Final test-set distribution/calibration metrics (Test A, Test B,
      Coverage) from the SLURM .out logs -> summary table + CSV + LaTeX
  (G) Case-study forecast curves + Test A/B distribution comparison for
      1-2 selected plants per config (from pre-generated Plotly HTML) ->
      figures + coverage table (CSV + LaTeX)

Run:
    python aggregate_results_trafo.py            # all configs in CONFIGS
    python aggregate_results_trafo.py C1         # only C1
    python aggregate_results_trafo.py C1 C4      # only C1 and C4
"""

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) EDIT THIS: configurations, abbreviations, job IDs (ONE per config)
# ----------------------------------------------------------------------
BASE_DIR = Path("output/pv_timeseries_tft_v2")
CSV_NAME = "per_plant_test_results.csv"

CONFIGS = {
    "C1": {"abbrev": "LSTM-Trafo-Base",   "job_id": "367506"},
    "C2": {"abbrev": "TFT-Trafo-Base",    "job_id": "368500"},
    "C3": {"abbrev": "LSTM-Trafo-FOMAML", "job_id": "367503"},
    "C4": {"abbrev": "TFT-Trafo-FOMAML",  "job_id": "352433"},
    "C5": {"abbrev": "LSTM-Trafo-ANIL",   "job_id": "352819"},
    "C6": {"abbrev": "TFT-Trafo-ANIL",    "job_id": "352211"},
    "C7": {"abbrev": "LSTM-Trafo-NIL",    "job_id": "368580"},
    "C8": {"abbrev": "TFT-Trafo-NIL",     "job_id": "352434"},
}

# ----------------------------------------------------------------------
# Optional CLI filter: restrict to a subset of configs, e.g.
#   python aggregate_results_trafo.py C1
#   python aggregate_results_trafo.py C1 C4
# ----------------------------------------------------------------------
if len(sys.argv) > 1:
    requested = sys.argv[1:]
    unknown = [c for c in requested if c not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown config id(s): {unknown}. Available: {list(CONFIGS.keys())}")
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in requested}
    print(f"[info] Running only: {list(CONFIGS.keys())}")

# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------
OUT_DIR = Path("/mnt/data/kes/data/jdomingo/personal-new/thesis/results_trafo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CASE_STUDY_DIR = OUT_DIR / "case_studies"
CASE_STUDY_DIR.mkdir(parents=True, exist_ok=True)

# Final test-metric logs (.out files)
OUT_LOG_ROOT = Path("output/logs")
OUT_LOG_FILE_GLOB_TMPL = "*_{job_id}.out"
OUT_LOG_TAIL_CHUNK = 65536
OUT_LOG_TAIL_MAX_BYTES = 5_000_000

FINAL_METRIC_KEYS = {
    "ks_A_mean": "test/ks_A_mean",
    "ks_B_mean": "test/ks_B_mean",
    "coverage_mean": "test/coverage_mean",
}

# ----------------------------------------------------------------------
# Part G config: Case-study forecast curves from pre-generated Plotly HTML
#   output/<unknown_folder>/prediction_plots/job_<job_id>/
# ----------------------------------------------------------------------
PRED_PLOTS_SEARCH_ROOT = Path("output")
PRED_PLOTS_SUBDIR = "prediction_plots"
PRED_PLOTS_SUMMARY_NAME = "prediction_summary.csv"

# How many scenarios to pick per config, and which groups to prefer
# (in order of preference). Typically prediction_summary.csv contains
# groups like "best" / "middle" / "worst" (one plant each, or more).
N_CASE_STUDY_SCENARIOS = 2
PREFERRED_GROUPS = ["best", "worst", "middle"]

_PLOTLY_DTYPE_MAP = {
    "f8": np.float64, "f4": np.float32,
    "i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64,
    "u1": np.uint8, "u2": np.uint16, "u4": np.uint32, "u8": np.uint64,
}


# ═══════════════════════════════════════════════════════════════════
# PART A — Per-plant MAE (single seed, no averaging)
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


def distribution_stats(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "std": series.std(ddof=1),
    }


def to_latex_mae_stats(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & {row['Phase']} & "
            f"{row['mean']:.4f} & {row['median']:.4f} & {row['min']:.4f} & "
            f"{row['max']:.4f} & {row['std']:.4f} \\\\"
        )
    return "\n".join(lines)


def to_latex_mae_summary(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['ZeroShot_MAE']:.4f} & {row['Adapted_MAE']:.4f} \\\\"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PART E — Final test-set distribution/calibration metrics
#          (ks_A_mean, ks_B_mean, coverage_mean) from SLURM .out logs
# ═══════════════════════════════════════════════════════════════════

_out_log_path_cache: Dict[str, Path] = {}


def find_out_log_file(job_id: str) -> Path:
    if job_id in _out_log_path_cache:
        return _out_log_path_cache[job_id]

    if not OUT_LOG_ROOT.is_dir():
        raise FileNotFoundError(f"Log root not found: {OUT_LOG_ROOT}")

    pattern = OUT_LOG_FILE_GLOB_TMPL.format(job_id=job_id)
    matches = list(OUT_LOG_ROOT.glob(pattern))
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
    required = list(metric_keys.values())
    text = tail_read_text(path, required_substrings=required)

    results = {}
    for short_name, log_key in metric_keys.items():
        pattern = re.compile(
            rf"{re.escape(log_key)}\s*[│|]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        results[short_name] = float(matches[-1]) if matches else np.nan
        if not matches:
            print(f"  [warn] '{log_key}' not found in tail of {path.name}")

    return results


def to_latex_final_metrics(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{row['Config']} & {row['Abbreviation']} & "
            f"{row['ks_A_mean']:.4f} & {row['ks_B_mean']:.4f} & "
            f"{row['coverage_mean']:.2f} \\\\"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PART G — Case-study forecast curves (1-2 scenarios per config) from
#          pre-generated Plotly HTML (PredictionPlotEvaluator output)
# ═══════════════════════════════════════════════════════════════════

_pred_plots_job_dir_cache: Dict[str, Path] = {}


def find_job_dir_for_prediction_plots(job_id: str) -> Path:
    """
    Locate output/<unknown_folder>/prediction_plots/job_<job_id>/.
    Only the top-level folder name under output/ is unknown, so only that
    level is scanned (one non-recursive iterdir()).
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
    Parses the LAST `Plotly.newPlot("divid", DATA, LAYOUT, CONFIG)` call in
    a Plotly fig.write_html() output (NOT the first occurrence, which can
    be inside the embedded plotly.js library's own source string).
    """
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    marker = "Plotly.newPlot("
    start = text.rfind(marker)
    if start == -1:
        raise ValueError(f"No 'Plotly.newPlot(' call found in {html_path.name}")
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
    qlines = _find_trace(data, name_regex=r"^q\d*\.?\d+$")

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
    return {"x_support": x_sup, "y_support": y_sup}


def load_case_study_plant(job_dir: Path, group: str, plant_id: str) -> dict:
    base = f"{group}_plant_{plant_id}"
    query = load_query_arrays(job_dir / f"{base}_pred_vs_true.html")
    support_path = job_dir / f"{base}_support.html"
    support = load_support_arrays(support_path) if support_path.exists() else {
        "x_support": None, "y_support": np.array([])
    }
    return {**query, **support, "plant_id": plant_id, "group": group}


def select_case_study_scenarios(summary_df: pd.DataFrame,
                                 n: int = N_CASE_STUDY_SCENARIOS,
                                 preferred_groups: List[str] = PREFERRED_GROUPS
                                 ) -> Dict[str, str]:
    """
    Picks up to `n` scenarios from the available groups in summary_df,
    preferring groups in `preferred_groups` order (e.g. "best" then
    "worst"). Returns dict: label -> plant_id, with the group looked up
    separately. Label is the group name (capitalized), e.g. "Best", "Worst".
    """
    picks = {}
    available_groups = summary_df["group"].unique().tolist()
    ordered_groups = [g for g in preferred_groups if g in available_groups]
    ordered_groups += [g for g in available_groups if g not in ordered_groups]

    for group in ordered_groups[:n]:
        grp = summary_df[summary_df["group"] == group]
        if grp.empty:
            continue
        # representative plant: lowest MAE for "best"-like groups, else median
        if group == "best":
            pid = grp.loc[grp["MAE"].idxmin(), "plant_id"]
        elif group == "worst":
            pid = grp.loc[grp["MAE"].idxmax(), "plant_id"]
        else:
            med = grp["MAE"].median()
            pid = grp.loc[(grp["MAE"] - med).abs().idxmin(), "plant_id"]
        picks[group.capitalize()] = str(pid)

    return picks


def short_plant_id(pid, n: int = 3) -> str:
    s = str(pid)
    if s.endswith(".0"):
        s = s[:-2]
    return f"...{s[-n:]}" if len(s) > n else s


def plot_case_study_grid(selected: Dict[str, dict], out_path: Path, config_label: str):
    """Grid figure: one row per scenario, support (left) | query forecast (right)."""
    criteria = list(selected.keys())
    n = len(criteria)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)

    for i, crit in enumerate(criteria):
        d = selected[crit]
        ax_sup, ax_qry = axes[i, 0], axes[i, 1]

        y_sup = d.get("y_support", np.array([]))
        if y_sup.size:
            x_sup_plot = np.arange(len(y_sup)) if d.get("x_support") is None else d["x_support"]
            ax_sup.plot(x_sup_plot, y_sup, color="darkorange", linewidth=1, label="support (measured)")
        ax_sup.set_title(f"{crit}: Plant {short_plant_id(d['plant_id'])} — support", fontsize=9)
        ax_sup.grid(alpha=0.3)
        ax_sup.legend(fontsize=7)

        x_q = d["x"] if d["x"] is not None else np.arange(len(d["y_true"]))
        if d["q_lo"].size and d["q_hi"].size:
            ax_qry.fill_between(x_q, d["q_lo"], d["q_hi"], alpha=0.2, color="royalblue",
                                 label=r"$[q_{0.1}, q_{0.9}]$")
        ax_qry.plot(x_q, d["y_median"], color="royalblue", linewidth=1.3, label="median prediction")
        ax_qry.plot(x_q, d["y_true"], color="black", linewidth=1, label="measured")
        ax_qry.set_title(f"{crit}: Plant {short_plant_id(d['plant_id'])} — query", fontsize=9)
        ax_qry.grid(alpha=0.3)
        ax_qry.legend(fontsize=7)

    fig.suptitle(f"Case-study forecast curves — {config_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [figure] saved -> {out_path}")


def plot_case_study_distributions(selected: Dict[str, dict], out_path_shift: Path,
                                   out_path_calib: Path, config_label: str,
                                   threshold: float = 0.0):
    """
    Test A (covariate shift)  : support vs. query distribution per scenario
    Test B (calibration)      : predicted median vs. query distribution per scenario
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
        ax.set_title(f"{crit}\nPlant {short_plant_id(d['plant_id'])} — Test A", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = axes_b[0, i]
        for arr, lbl, color in [(y_med, "pred. median", "#4CAF50"), (y_true, "query", "#FF5722")]:
            if arr.size:
                sv = np.sort(arr)
                p = np.arange(1, len(sv) + 1) / len(sv)
                ax2.plot(sv, p, label=f"{lbl} (n={len(arr)})", color=color, linewidth=1.5)
        ax2.set_title(f"{crit}\nPlant {short_plant_id(d['plant_id'])} — Test B", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

    fig_a.suptitle(f"Test A — Covariate shift (support vs. query) — {config_label}")
    fig_a.tight_layout()
    fig_a.savefig(out_path_shift, dpi=200)
    plt.close(fig_a)
    print(f"  [figure] saved -> {out_path_shift}")

    fig_b.suptitle(f"Test B — Calibration (predicted median vs. query) — {config_label}")
    fig_b.tight_layout()
    fig_b.savefig(out_path_calib, dpi=200)
    plt.close(fig_b)
    print(f"  [figure] saved -> {out_path_calib}")


def compute_case_coverage(selected: Dict[str, dict], threshold: float = 0.0) -> pd.DataFrame:
    """% of query targets inside [q_lo, q_hi], per scenario."""
    rows = []
    for crit, d in selected.items():
        mask = d["y_true"] > threshold
        y = d["y_true"][mask]
        lo = d["q_lo"][mask] if d["q_lo"].size else None
        hi = d["q_hi"][mask] if d["q_hi"].size else None
        if lo is None or hi is None or len(y) == 0:
            cov = np.nan
        else:
            cov = 100.0 * np.mean((y >= lo) & (y <= hi))
        rows.append({"Scenario": crit, "plant_id": d["plant_id"], "Coverage_pct": cov})
    return pd.DataFrame(rows)


def to_latex_case_coverage(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        cov = f"{row['Coverage_pct']:.2f}" if pd.notna(row["Coverage_pct"]) else "--"
        lines.append(f"{row['Scenario']} & {short_plant_id(row['plant_id'])} & {cov} \\\\")
    return "\n".join(lines)


def run_case_study_analysis(config_id: str, abbrev: str, job_id: str):
    """
    Section-6.5-style driver, single seed: resolves 1-2 scenarios (best/
    worst by default) and builds the forecast-curve grid + Test A/B figures
    + coverage table for this config.
    """
    print(f"\n[info] Case studies for {config_id} ({abbrev}), job_id={job_id}")

    try:
        job_dir = find_job_dir_for_prediction_plots(job_id)
        summary_df = load_prediction_summary(job_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip case studies, {config_id}] {e}")
        return

    picks = select_case_study_scenarios(summary_df)
    if not picks:
        print(f"  [skip case studies, {config_id}] no scenarios could be selected.")
        return

    selected = {}
    for label, pid in picks.items():
        group = label.lower()
        try:
            selected[label] = load_case_study_plant(job_dir, group, pid)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [warn] could not load scenario {label} ({pid}): {e}")

    if not selected:
        print(f"  [skip case studies, {config_id}] no plant data could be loaded.")
        return

    config_label = f"{config_id} ({abbrev})"
    plot_case_study_grid(selected, FIGURES_DIR / f"{config_id}_case_curves.pdf", config_label)
    plot_case_study_distributions(
        selected,
        out_path_shift=FIGURES_DIR / f"{config_id}_case_covariate_shift.pdf",
        out_path_calib=FIGURES_DIR / f"{config_id}_case_calibration.pdf",
        config_label=config_label,
    )

    coverage_df = compute_case_coverage(selected)
    coverage_df.to_csv(CASE_STUDY_DIR / f"{config_id}_case_coverage.csv", index=False)
    latex_cov = to_latex_case_coverage(coverage_df)
    (CASE_STUDY_DIR / f"{config_id}_case_coverage.tex").write_text(latex_cov)

    print(f"\n################ Table: case_coverage ({config_id}) ################\n")
    print(coverage_df.to_string(index=False))
    print(f"\n################ LaTeX body: case_coverage ({config_id}) ################\n")
    print(latex_cov)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    stats_rows = []       # -> Table "aggregate_stats" (per-plant MAE distribution)
    perf_rows = []        # -> Table "aggregate_performance" (single scalar per config)
    final_metrics_rows = []  # -> Table "final_test_metrics"

    for config_id, meta in CONFIGS.items():
        abbrev = meta["abbrev"]
        job_id = meta["job_id"]
        print(f"\n{'='*80}\n{config_id} ({abbrev}) : job_id {job_id}\n{'='*80}")

        # ---------- PART A: Zero-shot / Adapted MAE (single seed) ----------
        try:
            df = load_job_csv(job_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip MAE] {e}")
            df = None

        if df is not None:
            zs_stats = distribution_stats(df["zero_shot_MAE"])
            ad_stats = distribution_stats(df["adapted_MAE"])

            stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Zero-Shot", **zs_stats})
            stats_rows.append({"Config": config_id, "Abbreviation": abbrev, "Phase": "Adapted", **ad_stats})

            perf_rows.append({
                "Config": config_id,
                "Abbreviation": abbrev,
                "ZeroShot_MAE": zs_stats["mean"],
                "Adapted_MAE": ad_stats["mean"],
            })

            print(f"  [MAE] Zero-Shot : {zs_stats}")
            print(f"  [MAE] Adapted   : {ad_stats}")

        # ---------- PART E: Final test-set distribution/calibration metrics ----------
        try:
            log_path = find_out_log_file(job_id)
            metrics = parse_final_test_metrics(log_path)
            final_metrics_rows.append({"Config": config_id, "Abbreviation": abbrev, **metrics})
            print(f"  [final metrics] {metrics}")
        except FileNotFoundError as e:
            print(f"  [skip final metrics] {e}")

        # ---------- PART G: Case-study forecast curves (1-2 scenarios) ----------
        run_case_study_analysis(config_id, abbrev, job_id)

    # ---------------- Assemble dataframes ----------------
    stats_df = pd.DataFrame(stats_rows)
    perf_df = pd.DataFrame(perf_rows)
    final_metrics_df = pd.DataFrame(final_metrics_rows)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n\n################ Table: aggregate_stats (per-plant MAE distribution) ################\n")
    print(stats_df.to_string(index=False) if not stats_df.empty else "(no data)")

    print("\n\n################ Table: aggregate_performance (Zero-Shot vs. Adapted MAE) ################\n")
    print(perf_df.to_string(index=False) if not perf_df.empty else "(no data)")

    print("\n\n################ Table: final_test_metrics (Test A, Test B, Coverage) ################\n")
    print(final_metrics_df.to_string(index=False) if not final_metrics_df.empty else "(no data)")

    # ---------------- Save CSVs ----------------
    stats_df.to_csv(OUT_DIR / "aggregate_stats.csv", index=False)
    perf_df.to_csv(OUT_DIR / "aggregate_performance_mae.csv", index=False)
    final_metrics_df.to_csv(OUT_DIR / "final_test_metrics.csv", index=False)

    # ---------------- Save LaTeX bodies ----------------
    latex_stats = to_latex_mae_stats(stats_df) if not stats_df.empty else "% no data"
    latex_perf = to_latex_mae_summary(perf_df) if not perf_df.empty else "% no data"
    latex_final = to_latex_final_metrics(final_metrics_df) if not final_metrics_df.empty else "% no data"

    (OUT_DIR / "aggregate_stats.tex").write_text(latex_stats)
    (OUT_DIR / "aggregate_performance_mae.tex").write_text(latex_perf)
    (OUT_DIR / "final_test_metrics.tex").write_text(latex_final)

    print("\n\n################ LaTeX body: aggregate_stats ################\n")
    print(latex_stats)

    print("\n\n################ LaTeX body: aggregate_performance ################\n")
    print(latex_perf)

    print("\n\n################ LaTeX body: final_test_metrics ################\n")
    print(latex_final)

    print(f"\nSaved CSVs/LaTeX tables to: {OUT_DIR}/")
    print(f"Case-study figures/tables saved to: {FIGURES_DIR}/ and {CASE_STUDY_DIR}/")


if __name__ == "__main__":
    main()