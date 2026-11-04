# """
# Parse a MAML/ANIL .err log file and produce:
#   1. Per-task table with zero-shot query, zero-shot support, after-1-step,
#      after-last-step, query_loss, and generalization error columns.
#   2. Summary statistics (mean, min, max, quantiles).
#   3. Negative generalization error counts and percentages.

# Usage:
#     python utils/analyze_log.py output/logs/pf_tft_216293.err
# """

# import re
# import sys
# import numpy as np
# import pandas as pd
# from pathlib import Path


# # ═══════════════════════════════════════════════════════════════════════════════
# # PARSING
# # ═══════════════════════════════════════════════════════════════════════════════

# def parse_log_file(filepath: str) -> pd.DataFrame:
#     """Parse .err log and return a per-task DataFrame."""

#     step_re = re.compile(
#         r"\[task\s+(\d+)\s*\|\s*step\s+(\d+)\]\s*train_loss=([\d.]+)"
#     )
#     # query_re = re.compile(
#     #     r"\[task\s+(\d+)\s*\|\s*query\]\s*test_loss=([\d.]+)"
#     # )
#     # query_re = re.compile(
#     #     r"\[task\s+(\d+)\s*\|\s*query\]\s*test_loss=([\d.]+)"
#     #     r"(?:\s*\|\s*masked=(nan|[-\d.eE+]+))?"
#     # )

#     query_re = re.compile(
#         r"\[task\s+(\d+)\s*\|\s*query\]\s*test_loss=([\d.]+)"
#         r"(?:\s*\|\s*masked=(nan|[-\d.eE+]+))?"
#         r"(?:\s*\|\s*max_loss=(nan|[-\d.eE+]+))?"   # ← new
#     )

#     zero_query_re = re.compile(
#         r"\[task\s+(\d+)\s*\|\s*zero-shot query\]\s*test_loss=([\d.]+)"
#     )
    
#     zero_query_re = re.compile(
#     r"\[task\s+(\d+)\s*\|\s*zero-shot query\]\s*test_loss=([\d.]+)"
#     r"(?:\s*\|\s*masked=(nan|[-\d.eE+]+))?"   # ← add this
#     )

#     status_re = re.compile(r"Current training status:\s*(True|False)")

#     tasks = []
#     current = None
#     global_id = 0
#     segment_id = 0

#     with open(filepath, "r") as f:
#         for line in f:
#             m_zero_query = zero_query_re.search(line)
#             m_step = step_re.search(line)
#             m_query = query_re.search(line)

#             m_status = status_re.search(line)
#             if m_status:
#                 if current is not None:
#                     tasks.append(current)
#                     current = None
#                 segment_id += 1     # init segment_id = 0 before the loop
#                 continue

#             if m_zero_query:
#                 # zero-shot query appears BEFORE step 0 for that task
#                 task_id = int(m_zero_query.group(1))
#                 zero_q_loss = float(m_zero_query.group(2))
#                 masked_str = m_zero_query.group(3)


#                 # Save previous task if exists
#                 if current is not None:
#                     tasks.append(current)

#                 current = {
#                     "task_global_id": global_id,
#                     "task_local_id": task_id,
#                     "segment_id": segment_id,
#                     # "zero_shot_query": zero_q_loss,
#                     "zero_shot_query":        float(m_zero_query.group(2)),
#                     "zero_shot_query_masked": (
#                         np.nan if masked_str is None or masked_str == "nan" else float(masked_str)),
#                     "steps": {},
#                 }
#                 global_id += 1

#             elif m_step:
#                 task_id = int(m_step.group(1))
#                 step = int(m_step.group(2))
#                 loss = float(m_step.group(3))

#                 if step == 0 and current is None:
#                     # Fallback: no zero-shot query line exists (old log format)
#                     current = {
#                         "task_global_id": global_id,
#                         "task_local_id": task_id,
#                         "segment_id": segment_id,
#                         "zero_shot_query": np.nan,
#                         "steps": {},
#                     }
#                     global_id += 1
#                 elif step == 0 and current is not None and "steps" in current and len(current["steps"]) > 0:
#                     # New task started without zero-shot query (old format)
#                     tasks.append(current)
#                     current = {
#                         "task_global_id": global_id,
#                         "task_local_id": task_id,
#                         "segment_id": segment_id,
#                         "zero_shot_query": np.nan,
#                         "steps": {},
#                     }
#                     global_id += 1

#                 if current is not None:
#                     current["steps"][step] = loss

#             elif m_query:
#                 query_loss   = float(m_query.group(2))
#                 masked_str   = m_query.group(3)
#                 max_loss_str = m_query.group(4)            # ← new
#                 if current is not None:
#                     current["query_loss"] = query_loss
#                     if masked_str is not None:
#                         current["query_loss_masked"] = (
#                             np.nan if masked_str == "nan" else float(masked_str)
#                         )
#                     if max_loss_str is not None:           # ← new
#                         current["query_loss_max"] = (
#                             np.nan if max_loss_str == "nan" else float(max_loss_str)
#                         )
#     # Append last task
#     if current is not None:
#         tasks.append(current)

#     # Build records
#     records = []
#     for t in tasks:
#         steps = t["steps"]
#         sorted_keys = sorted(steps.keys())
#         last_step = sorted_keys[-1] if sorted_keys else None

#         records.append({
#             "task_global_id": t["task_global_id"],
#             "task_local_id": t["task_local_id"],
#             "segment_id": t["segment_id"],
#             "zero_shot_query": t.get("zero_shot_query", np.nan),
#             "zero_shot_support": steps.get(0, np.nan),
#             "after_1_step": steps.get(1, np.nan),
#             "after_last_step": steps.get(last_step, np.nan) if last_step is not None else np.nan,
#             "n_adaptation_steps": (last_step + 1) if last_step is not None else 0,
#             "query_loss": t.get("query_loss", np.nan),
#             "query_loss_max": t.get("query_loss_max", np.nan),
#             "query_loss_masked": t.get("query_loss_masked", np.nan),
#             "zero_shot_query_masked": t.get("zero_shot_query_masked", np.nan),

#         })

#     df = pd.DataFrame(records)

#     # Generalization errors: query_loss − train_loss_at_step
#     df["gen_error_vs_zero"] = df["query_loss"] - df["zero_shot_support"]
#     df["gen_error_vs_1step"] = df["query_loss"] - df["after_1_step"]
#     df["gen_error_vs_last"] = df["query_loss"] - df["after_last_step"]

#     # Improvement from zero-shot query to final query (positive = improved)
#     df["query_improvement"] = df["zero_shot_query"] - df["query_loss"]

#     return df


# # ═══════════════════════════════════════════════════════════════════════════════
# # SUMMARY STATISTICS
# # ═══════════════════════════════════════════════════════════════════════════════

# def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
#     """Mean, min, max, std, and quantiles across all tasks."""

#     cols = [
#         "zero_shot_query", "zero_shot_support", "after_1_step",
#         "after_last_step", "query_loss", "query_loss_max", "query_loss_masked", "zero_shot_query_masked",
#         "gen_error_vs_zero", "gen_error_vs_1step", "gen_error_vs_last",
#         "query_improvement",
#     ]

#     rows = []
#     for col in cols:
#         if col not in df.columns:
#             continue
#         s = df[col].dropna()
#         if len(s) == 0:
#             continue
#         rows.append({
#             "metric": col,
#             "mean": s.mean(),
#             "std": s.std(),
#             "min": s.min(),
#             "q05": s.quantile(0.05),
#             "q10": s.quantile(0.1),
#             "median": s.quantile(0.5),
#             "q90": s.quantile(0.9),
#             "q95": s.quantile(0.95),
#             "max": s.max(),
#         })

#     return pd.DataFrame(rows)


# # ═══════════════════════════════════════════════════════════════════════════════
# # NEGATIVE GENERALIZATION ERROR
# # ═══════════════════════════════════════════════════════════════════════════════

# def compute_negative_gen_errors(df: pd.DataFrame) -> pd.DataFrame:
#     gen_cols = {
#         "gen_error_vs_zero":  "before adaptation  (step 0)",
#         "gen_error_vs_1step": "after  1 step",
#         "gen_error_vs_last":  "after  last step   (full adaptation)",
#     }
#     total = len(df)
#     rows = []
#     for col, label in gen_cols.items():
#         n_neg = (df[col] < 0).sum()
#         n_pos = total - n_neg
#         rows.append({
#             "stage": label,
#             "query < support (good generalization)": f"{n_neg:>6} ({100*n_neg/total:.1f}%)",
#             "query > support (potential overfit)":   f"{n_pos:>6} ({100*n_pos/total:.1f}%)",
#         })
#     return pd.DataFrame(rows)


# # ═══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ═══════════════════════════════════════════════════════════════════════════════

# def main():
#     filepath = sys.argv[1]

#     print(f"{'═' * 80}")
#     print(f"  MAML Inner-Loop Analysis — {Path(filepath).name}")
#     print(f"{'═' * 80}\n")

#     df = parse_log_file(filepath)
#     # df = df[(df["task_local_id"].diff() < 0).cumsum() >= 5].reset_index(drop=True)
#     # df = df[df["segment_id"] == df["segment_id"].min()].reset_index(drop=True)
    
#     # TASKS_PER_EPOCH = 6000   # 600 batches × 10 tasks

#     # df = df.head(TASKS_PER_EPOCH).reset_index(drop=True)     # first epoch
#     # ── [1] Per-task table ────────────────────────────────────────────────────
#     display_cols = [
#         "task_global_id", "task_local_id",
#         "zero_shot_query", "zero_shot_support", "after_1_step",
#         "after_last_step", "query_loss", "query_loss_masked","zero_shot_query_masked",
#         "gen_error_vs_zero", "gen_error_vs_1step", "gen_error_vs_last",
#         "query_improvement",
#     ]
#     # print("[1] Per-Task Table")
#     # print("-" * 80)
#     # print(df[display_cols].to_string(index=False, float_format="{:.6f}".format))

#     # ── [2] Summary statistics ────────────────────────────────────────────────
#     summary = compute_summary(df)
#     print(f"\n\n[2] Summary Statistics (N={len(df)} tasks)")
#     print("-" * 80)
#     print(summary.to_string(index=False, float_format="{:.6f}".format))

#     # ── [3] Negative generalization error ─────────────────────────────────────
#     neg_df = compute_negative_gen_errors(df)
#     print("\n[3] Generalization Gap Analysis")
#     print("    Negative = query_loss < support_loss → model generalises BETTER than it trains")
#     print("    Positive = query_loss > support_loss → model overfits to support set")
#     print("    Trend: gap should SHRINK as adaptation steps increase (✓ = healthy)")
#     print("-" * 90)
#     print(neg_df.to_string(index=False))
#     print("\n  Interpretation:")
#     print(f"  • Before adaptation : {neg_df.iloc[0]['query < support (good generalization)']} of tasks generalise well")
#     print(f"  • After last step   : {neg_df.iloc[2]['query < support (good generalization)']} — adaptation reduces overfit risk")

#     # ── Save CSVs ─────────────────────────────────────────────────────────────
#     out_dir = Path(filepath).parent
#     stem = Path(filepath).stem

#     df.to_csv(f"utils/output/{stem}_per_task.csv", index=False)
#     summary.to_csv(f"utils/output/{stem}_summary.csv", index=False)
#     neg_df.to_csv(f"utils/output/{stem}_neg_gen_error.csv", index=False)

#     print(f"\n\n✓ CSVs saved to: utils/output/")
#     print(f"  • {stem}_per_task.csv")
#     print(f"  • {stem}_summary.csv")
#     print(f"  • {stem}_neg_gen_error.csv")


# if __name__ == "__main__":
#     main()


"""
Parse a MAML/ANIL .err log file and produce:
  1. Per-task table with zero-shot query, zero-shot support, after-1-step,
     after-last-step, query_loss, and generalization error columns.
  2. Summary statistics (mean, min, max, quantiles).
  3. Negative generalization error counts and percentages.
  4. Full inner-loop step curve (adaptation speed) — mean/median loss per step.
  5. Steps-to-threshold convergence metric (per task).

Usage:
    python utils/analyze_log.py output/logs/pf_tft_216293.err
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_log_file(filepath: str):
    """
    Parse .err log and return:
        df    -> per-task DataFrame (zero-shot / step-0 / step-1 / last-step / query)
        tasks -> raw list of task dicts, each containing the FULL "steps" dict
                 {step_idx: loss}, used later to build the adaptation speed curve.
    """

    step_re = re.compile(
        r"\[task\s+(\d+)\s*\|\s*step\s+(\d+)\]\s*train_loss=([\d.]+)"
    )

    query_re = re.compile(
        r"\[task\s+(\d+)\s*\|\s*query\]\s*test_loss=([\d.]+)"
        r"(?:\s*\|\s*masked=(nan|[-\d.eE+]+))?"
        r"(?:\s*\|\s*max_loss=(nan|[-\d.eE+]+))?"
    )

    zero_query_re = re.compile(
        r"\[task\s+(\d+)\s*\|\s*zero-shot query\]\s*test_loss=([\d.]+)"
        r"(?:\s*\|\s*masked=(nan|[-\d.eE+]+))?"
    )

    status_re = re.compile(r"Current training status:\s*(True|False)")

    tasks = []
    current = None
    global_id = 0
    segment_id = 0

    with open(filepath, "r") as f:
        for line in f:
            m_zero_query = zero_query_re.search(line)
            m_step = step_re.search(line)
            m_query = query_re.search(line)
            m_status = status_re.search(line)

            if m_status:
                if current is not None:
                    tasks.append(current)
                    current = None
                segment_id += 1
                continue

            if m_zero_query:
                task_id = int(m_zero_query.group(1))
                zero_q_loss = float(m_zero_query.group(2))
                masked_str = m_zero_query.group(3)

                if current is not None:
                    tasks.append(current)

                current = {
                    "task_global_id": global_id,
                    "task_local_id": task_id,
                    "segment_id": segment_id,
                    "zero_shot_query": zero_q_loss,
                    "zero_shot_query_masked": (
                        np.nan if masked_str is None or masked_str == "nan"
                        else float(masked_str)
                    ),
                    "steps": {},
                }
                global_id += 1

            elif m_step:
                task_id = int(m_step.group(1))
                step = int(m_step.group(2))
                loss = float(m_step.group(3))

                if step == 0 and current is None:
                    current = {
                        "task_global_id": global_id,
                        "task_local_id": task_id,
                        "segment_id": segment_id,
                        "zero_shot_query": np.nan,
                        "zero_shot_query_masked": np.nan,
                        "steps": {},
                    }
                    global_id += 1
                elif step == 0 and current is not None and len(current["steps"]) > 0:
                    tasks.append(current)
                    current = {
                        "task_global_id": global_id,
                        "task_local_id": task_id,
                        "segment_id": segment_id,
                        "zero_shot_query": np.nan,
                        "zero_shot_query_masked": np.nan,
                        "steps": {},
                    }
                    global_id += 1

                if current is not None:
                    current["steps"][step] = loss

            elif m_query:
                query_loss = float(m_query.group(2))
                masked_str = m_query.group(3)
                max_loss_str = m_query.group(4)
                if current is not None:
                    current["query_loss"] = query_loss
                    if masked_str is not None:
                        current["query_loss_masked"] = (
                            np.nan if masked_str == "nan" else float(masked_str)
                        )
                    if max_loss_str is not None:
                        current["query_loss_max"] = (
                            np.nan if max_loss_str == "nan" else float(max_loss_str)
                        )

    if current is not None:
        tasks.append(current)

    # Build per-task summary records (step 0 / 1 / last)
    records = []
    for t in tasks:
        steps = t["steps"]
        sorted_keys = sorted(steps.keys())
        last_step = sorted_keys[-1] if sorted_keys else None

        records.append({
            "task_global_id": t["task_global_id"],
            "task_local_id": t["task_local_id"],
            "segment_id": t["segment_id"],
            "zero_shot_query": t.get("zero_shot_query", np.nan),
            "zero_shot_support": steps.get(0, np.nan),
            "after_1_step": steps.get(1, np.nan),
            "after_last_step": steps.get(last_step, np.nan) if last_step is not None else np.nan,
            "n_adaptation_steps": (last_step + 1) if last_step is not None else 0,
            "query_loss": t.get("query_loss", np.nan),
            "query_loss_max": t.get("query_loss_max", np.nan),
            "query_loss_masked": t.get("query_loss_masked", np.nan),
            "zero_shot_query_masked": t.get("zero_shot_query_masked", np.nan),
        })

    df = pd.DataFrame(records)

    # Generalization errors: query_loss − train_loss_at_step
    df["gen_error_vs_zero"] = df["query_loss"] - df["zero_shot_support"]
    df["gen_error_vs_1step"] = df["query_loss"] - df["after_1_step"]
    df["gen_error_vs_last"] = df["query_loss"] - df["after_last_step"]

    # Improvement from zero-shot query to final query (positive = improved)
    df["query_improvement"] = df["zero_shot_query"] - df["query_loss"]

    return df, tasks


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, min, max, std, and quantiles across all tasks."""

    cols = [
        "zero_shot_query", "zero_shot_support", "after_1_step",
        "after_last_step", "query_loss", "query_loss_max", "query_loss_masked",
        "zero_shot_query_masked",
        "gen_error_vs_zero", "gen_error_vs_1step", "gen_error_vs_last",
        "query_improvement",
    ]

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "metric": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q05": s.quantile(0.05),
            "q10": s.quantile(0.1),
            "median": s.quantile(0.5),
            "q90": s.quantile(0.9),
            "q95": s.quantile(0.95),
            "max": s.max(),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE GENERALIZATION ERROR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negative_gen_errors(df: pd.DataFrame) -> pd.DataFrame:
    gen_cols = {
        "gen_error_vs_zero":  "before adaptation  (step 0)",
        "gen_error_vs_1step": "after  1 step",
        "gen_error_vs_last":  "after  last step   (full adaptation)",
    }
    total = len(df)
    rows = []
    for col, label in gen_cols.items():
        n_neg = (df[col] < 0).sum()
        n_pos = total - n_neg
        rows.append({
            "stage": label,
            "query < support (good generalization)": f"{n_neg:>6} ({100*n_neg/total:.1f}%)",
            "query > support (potential overfit)":   f"{n_pos:>6} ({100*n_pos/total:.1f}%)",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL INNER-LOOP STEP CURVE  (long-format: one row per task × step)
# ═══════════════════════════════════════════════════════════════════════════════

def build_step_curve_df(tasks: list) -> pd.DataFrame:
    """
    Build a long-format DataFrame: one row per (task, step).
    Preserves the FULL inner-loop trajectory, not just step 0/1/last.

    Also computes `frac_converged`: fraction of the TOTAL improvement
    (initial_loss - final_loss) already achieved at a given step.
    """
    rows = []
    for t in tasks:
        steps = t["steps"]
        if not steps:
            continue
        sorted_keys = sorted(steps.keys())
        n_steps = len(sorted_keys)
        initial_loss = steps[sorted_keys[0]]
        final_loss = steps[sorted_keys[-1]]
        total_improvement = initial_loss - final_loss

        for step_idx in sorted_keys:
            loss = steps[step_idx]
            if total_improvement != 0:
                frac_converged = (initial_loss - loss) / total_improvement
            else:
                frac_converged = np.nan

            rows.append({
                "task_global_id": t["task_global_id"],
                "task_local_id": t["task_local_id"],
                "segment_id": t["segment_id"],
                "step": step_idx,
                "train_loss": loss,
                "n_total_steps": n_steps,
                "frac_converged": frac_converged,
            })

    return pd.DataFrame(rows)


def compute_step_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate loss statistics PER STEP across all tasks.
    This is the actual 'adaptation speed curve'.
    """
    if step_df.empty:
        return pd.DataFrame()

    grouped = step_df.groupby("step")["train_loss"]
    summary = grouped.agg(
        mean="mean",
        std="std",
        median="median",
        q10=lambda s: s.quantile(0.1),
        q90=lambda s: s.quantile(0.9),
        n_tasks="count",
    ).reset_index()
    return summary


def compute_steps_to_threshold(step_df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    For each task, find the FIRST step at which frac_converged >= threshold.
    A single scalar 'speed' value per task: lower = faster adaptation.
    """
    if step_df.empty:
        return pd.DataFrame()

    results = []
    for task_id, g in step_df.groupby("task_global_id"):
        g_sorted = g.sort_values("step")
        reached = g_sorted[g_sorted["frac_converged"] >= threshold]
        step_reached = reached.iloc[0]["step"] if len(reached) > 0 else np.nan

        results.append({
            "task_global_id": task_id,
            "task_local_id": g_sorted.iloc[0]["task_local_id"],
            f"steps_to_{int(threshold*100)}pct": step_reached,
            "n_total_steps": g_sorted.iloc[0]["n_total_steps"],
        })

    return pd.DataFrame(results)


def plot_adaptation_curve(step_summary: pd.DataFrame, out_path: str):
    """Plot mean loss vs. step with q10-q90 band."""
    if step_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(step_summary["step"], step_summary["mean"],
            marker="o", color="tab:blue", label="mean train_loss")
    ax.fill_between(step_summary["step"],
                     step_summary["q10"], step_summary["q90"],
                     alpha=0.2, color="tab:blue", label="q10–q90")

    ax.set_xlabel("Adaptation step")
    ax.set_ylabel("Train loss (support set)")
    ax.set_title("Adaptation Speed Curve (loss vs. inner-loop step)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    filepath = sys.argv[1]

    print(f"{'═' * 80}")
    print(f"  MAML Inner-Loop Analysis — {Path(filepath).name}")
    print(f"{'═' * 80}\n")

    df, tasks = parse_log_file(filepath)

    out_dir = Path("utils/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem

    # ── [1] Per-task table ────────────────────────────────────────────────────
    display_cols = [
        "task_global_id", "task_local_id",
        "zero_shot_query", "zero_shot_support", "after_1_step",
        "after_last_step", "query_loss", "query_loss_masked", "zero_shot_query_masked",
        "gen_error_vs_zero", "gen_error_vs_1step", "gen_error_vs_last",
        "query_improvement",
    ]
    # print("[1] Per-Task Table")
    # print("-" * 80)
    # print(df[display_cols].to_string(index=False, float_format="{:.6f}".format))

    # ── [2] Summary statistics ────────────────────────────────────────────────
    summary = compute_summary(df)
    print(f"\n\n[2] Summary Statistics (N={len(df)} tasks)")
    print("-" * 80)
    print(summary.to_string(index=False, float_format="{:.6f}".format))

    # ── [3] Negative generalization error ─────────────────────────────────────
    neg_df = compute_negative_gen_errors(df)
    print("\n[3] Generalization Gap Analysis")
    print("    Negative = query_loss < support_loss → model generalises BETTER than it trains")
    print("    Positive = query_loss > support_loss → model overfits to support set")
    print("    Trend: gap should SHRINK as adaptation steps increase (✓ = healthy)")
    print("-" * 90)
    print(neg_df.to_string(index=False))
    print("\n  Interpretation:")
    print(f"  • Before adaptation : {neg_df.iloc[0]['query < support (good generalization)']} of tasks generalise well")
    print(f"  • After last step   : {neg_df.iloc[2]['query < support (good generalization)']} — adaptation reduces overfit risk")

    # ── [4] Full inner-loop step curve (ADAPTATION SPEED) ───────────────────────
    step_df = build_step_curve_df(tasks)
    step_summary = compute_step_summary(step_df)

    print(f"\n[4] Adaptation Speed Curve (per step, aggregated over all tasks)")
    print("-" * 80)
    if not step_summary.empty:
        print(step_summary.to_string(index=False, float_format="{:.6f}".format))
    else:
        print("  (no multi-step data found)")

    # ── [5] Steps-to-threshold convergence (per-task speed metric) ─────────────
    steps_to_90 = compute_steps_to_threshold(step_df, threshold=0.9)
    print(f"\n[5] Steps-to-90%-Convergence (per task)")
    print("-" * 80)
    if not steps_to_90.empty:
        print(steps_to_90["steps_to_90pct"].describe().to_string())
    else:
        print("  (no multi-step data found)")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df.to_csv(out_dir / f"{stem}_per_task.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    neg_df.to_csv(out_dir / f"{stem}_neg_gen_error.csv", index=False)
    step_df.to_csv(out_dir / f"{stem}_step_curve_long.csv", index=False)
    step_summary.to_csv(out_dir / f"{stem}_step_summary.csv", index=False)
    steps_to_90.to_csv(out_dir / f"{stem}_steps_to_90pct.csv", index=False)

    # ── Plot adaptation curve ───────────────────────────────────────────────
    plot_path = out_dir / f"{stem}_adaptation_curve.png"
    plot_adaptation_curve(step_summary, str(plot_path))

    print(f"\n\n✓ CSVs saved to: {out_dir}/")
    print(f"  • {stem}_per_task.csv")
    print(f"  • {stem}_summary.csv")
    print(f"  • {stem}_neg_gen_error.csv")
    print(f"  • {stem}_step_curve_long.csv")
    print(f"  • {stem}_step_summary.csv")
    print(f"  • {stem}_steps_to_90pct.csv")
    print(f"✓ Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()