"""
Sample Efficiency Curve — answers: "How much data does a new plant need?"

Reads results_cache from extended_inner_learning and plots:
    MAE vs. number of support windows (k-shot)

Each task in results_cache has metrics at:
    k=0 (zero-shot), k=1, k=2, ..., k=n_support
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any
import csv
from collections import defaultdict

import logging
log = logging.getLogger(__name__)


class SampleEfficiencyPlotter:
    """
    Plots the sample efficiency curve from meta-testing results.

    X-axis semantics:
        Primary  : k (number of support windows used for adaptation)
        Secondary: total hours of unique plant data at that k level

        At k=0: zero-shot (no plant data).
        At k=1: model sees 1 window = (look_back + look_ahead) timesteps.
        At k=n: sliding windows overlap by (window_steps - 1) steps,
                so total unique timesteps = window_steps + (n - 1).
                Total hours = (window_steps + n - 1) * step_minutes / 60.
    """

    def __init__(
        self,
        results_cache: List[List[Any]],
        save_path: str,
        look_back_steps: int = 288,
        look_ahead_steps: int = 96,
        step_minutes: float = 15.0,
        chunk_size: int = 1,
        window_stride: int = 1,
    ):
        """
        Args:
            results_cache : from LightningMAMLModule.results_cache
            save_path     : directory to save plots
            look_back_steps  : encoder window length in timesteps
            look_ahead_steps : forecast horizon in timesteps
            step_minutes  : duration of one timestep in minutes (15 for your data)
            chunk_size    : if chunks were used in extended_inner_learning
        """
        self.results_cache = results_cache
        self.save_path = save_path.rstrip("/") + "/"
        self.look_back_steps = look_back_steps
        self.look_ahead_steps = look_ahead_steps
        self.window_steps = look_back_steps + look_ahead_steps
        self.step_minutes = step_minutes
        self.chunk_size = chunk_size
        self.window_stride = window_stride
        os.makedirs(self.save_path, exist_ok=True)

    def _k_to_hours(self, k_values: np.ndarray) -> np.ndarray:
        """
        Convert k (number of support windows) to total hours of unique plant data.

        k=0 → 0 hours (zero-shot, no data)
        k≥1 → (window_steps + (k-1)*stride) * step_minutes / 60
            stride-aware: k windows with given stride cover
            window_steps + (k-1)*stride unique timesteps.
        """
        hours = np.zeros_like(k_values, dtype=float)
        mask = k_values >= 1
        hours[mask] = (
            (self.window_steps + (k_values[mask] - 1) * self.window_stride)
            * self.step_minutes / 60.0
        )
        return hours

    def _extract_curves(self):
        """
        Extract per-task k-shot MAE curves from results_cache.

        Structure: results_cache[task] = [metrics_k0, metrics_k1, ..., metrics_kN]
        Each metrics_kN["mae"]["test"] = [step0, step1, ..., final]
        
        We want the FINAL test MAE at each k → sample efficiency curve.
        """
        mae_curves = []

        print(f"[DEBUG] results_cache: {len(self.results_cache)} tasks")

        for t, task_results in enumerate(self.results_cache):
            if t < 3:  # only print first 3 tasks
                print(f"[DEBUG] task {t}: {len(task_results)} entries, "
                    f"type={type(task_results[0]) if task_results else 'empty'}")
                if task_results:
                    first = task_results[0]
                    print(f"[DEBUG]   keys={first.keys() if isinstance(first, dict) else 'not dict'}")
                    if isinstance(first, dict) and "mae" in first:
                        print(f"[DEBUG]   mae['test']={first['mae'].get('test', [])}")

            if not task_results or len(task_results) < 2:
                continue

            curve = []
            for metrics in task_results:
                if isinstance(metrics, dict) and "mae" in metrics:
                    test_mae_steps = metrics["mae"].get("test", [])
                    if test_mae_steps:
                        curve.append(test_mae_steps[-1])

            if len(curve) > 1:
                mae_curves.append(np.array(curve))

        print(f"[DEBUG] extracted {len(mae_curves)} curves, "
            f"lengths={[len(c) for c in mae_curves[:5]]}")

        return mae_curves

    def plot(self):
        """Generate all sample efficiency plots."""
        mae_curves = self._extract_curves()

        if not mae_curves:
            print("[SampleEfficiencyPlotter] No multi-step MAE data found in results_cache.")
            print("  Make sure sample_efficiency: true in metrics_config.")
            return

        print(f"\n[SampleEfficiencyPlotter] Found {len(mae_curves)} tasks with MAE curves.")

        # Pad to same length (tasks may have different n_support)
        max_len = max(len(c) for c in mae_curves)
        padded = np.full((len(mae_curves), max_len), np.nan)
        for i, c in enumerate(mae_curves):
            padded[i, :len(c)] = c

        self._plot_mean_curve(padded)
        self._plot_boxplot(padded)
        self._plot_individual_curves(padded)

        print("\n[SampleEfficiencyPlotter] Building statistics tables...")
        for metric in ["test_loss_masked", "test_mae_final"]:
            rows = self._build_stats_table(metric)
            self._print_stats_table(rows, metric)
            self._save_stats_csv(rows, metric)

    def _plot_mean_curve(self, padded: np.ndarray):
        """Main result: mean MAE ± std across tasks vs k-shot."""
        mean_mae = np.nanmean(padded, axis=0)
        std_mae = np.nanstd(padded, axis=0)
        n_steps = len(mean_mae)

        # Primary x-axis: number of support windows
        k_values = np.arange(n_steps) * self.chunk_size
        hours = self._k_to_hours(k_values)

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(k_values, mean_mae, color="steelblue", linewidth=2, marker="o",
                 markersize=4, label="Mean MAE")
        ax1.fill_between(k_values, mean_mae - std_mae, mean_mae + std_mae,
                         alpha=0.2, color="steelblue", label="±1 std")
        ax1.set_xlabel("k (number of support windows)")
        ax1.set_ylabel("MAE (normalized)")
        ax1.set_title(
            "Sample Efficiency: How Much Data Does a New Plant Need?\n"
            "(k-shot from meta-initialization — each k adapted independently)",
            fontsize=11,
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # Secondary axis: total hours of unique plant data
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        # Pick evenly spaced tick positions from the primary axis
        n_ticks = min(8, n_steps)
        tick_positions = np.linspace(0, k_values[-1], n_ticks)
        tick_hours = self._k_to_hours(tick_positions.astype(int))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f"{h:.1f}h" for h in tick_hours])
        ax2.set_xlabel("Total unique plant data (hours)")

        # Mark zero-shot baseline
        ax1.axhline(y=mean_mae[0], color="red", linestyle="--", alpha=0.5,
                    label=f"Zero-shot: {mean_mae[0]:.4f}")
        ax1.legend(loc="upper right")

        # Annotate elbow (largest drop)
        if n_steps > 2:
            drops = np.diff(mean_mae)
            elbow_idx = np.argmin(drops) + 1
            ax1.annotate(
                f"Biggest gain at k={k_values[elbow_idx]}\n"
                f"({hours[elbow_idx]:.1f}h of data)",
                xy=(k_values[elbow_idx], mean_mae[elbow_idx]),
                xytext=(k_values[elbow_idx] + max(2, n_steps * 0.1),
                        mean_mae[elbow_idx] + 0.02),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                fontsize=9, color="darkred",
            )

        plt.tight_layout()
        path = self.save_path + "sample_efficiency_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    def _plot_boxplot(self, padded: np.ndarray):
        """Box plot at selected k values — shows distribution across tasks."""
        n_steps = padded.shape[1]
        if n_steps <= 8:
            selected = list(range(n_steps))
        else:
            selected = [0, 1, 2, 4, 8, 12, 16, n_steps - 1]
            selected = [s for s in selected if s < n_steps]

        data = [padded[:, s][~np.isnan(padded[:, s])] for s in selected]
        k_vals = [s * self.chunk_size for s in selected]
        hours = self._k_to_hours(np.array(k_vals))
        labels = [f"k={k}\n({h:.1f}h)" for k, h in zip(k_vals, hours)]

        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")

        ax.set_xlabel("Support windows (k) and total plant data hours")
        ax.set_ylabel("MAE (normalized)")
        ax.set_title("MAE Distribution Across Test Plants at Each k-Shot Level")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = self.save_path + "sample_efficiency_boxplot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    def _plot_individual_curves(self, padded: np.ndarray, max_curves: int = 20):
        """Spaghetti plot: individual task curves (subsample if too many)."""
        n_tasks = padded.shape[0]
        n_steps = padded.shape[1]
        k_values = np.arange(n_steps) * self.chunk_size

        fig, ax = plt.subplots(figsize=(10, 5))

        indices = np.random.choice(n_tasks, size=min(max_curves, n_tasks), replace=False)

        for idx in indices:
            curve = padded[idx]
            valid = ~np.isnan(curve)
            ax.plot(k_values[valid], curve[valid], alpha=0.3, linewidth=0.8,
                    color="steelblue")

        mean_mae = np.nanmean(padded, axis=0)
        ax.plot(k_values, mean_mae, color="darkred", linewidth=2.5,
                label="Mean across tasks", zorder=10)

        ax.set_xlabel("k (number of support windows)")
        ax.set_ylabel("MAE (normalized)")
        ax.set_title(f"Individual Plant Adaptation Curves ({min(max_curves, n_tasks)} tasks)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.save_path + "sample_efficiency_individual.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    def _collect_by_k(self, metric: str = "test_loss_masked") -> dict:
        """
        Gather the metric values and group them after k-shot over all Tasks.
        metric = 'test_loss_masked' | 'test_mae_final'
        Returns: {k: [float, ...]}
        """
        k_data = defaultdict(list)
        for task in self.results_cache:
            for entry in task:
                k = entry.get("k_shot")
                if k is None:
                    continue
                if metric == "test_mae_final":
                    lst = entry.get("mae", {}).get("test", [])
                    val = lst[-1] if lst else None
                else:
                    val = entry.get(metric)
                if val is not None and not np.isnan(val):
                    k_data[k].append(float(val))
        return k_data


    def _build_stats_table(self, metric: str = "test_loss_masked") -> list[dict]:
        """
        Creates statistic rows per k-shot:
        mean, median, min, max, Q0.10, Q0.90
        """
        k_data = self._collect_by_k(metric)
        rows = []
        for k in sorted(k_data.keys()):
            vals = np.array(k_data[k])
            hours = self._k_to_hours(np.array([k]))[0]
            rows.append({
                "k_shot" : k,
                "hours"  : round(hours, 2),
                "n"      : len(vals),
                "mean"   : float(np.mean(vals)),
                "median" : float(np.median(vals)),
                "min"    : float(np.min(vals)),
                "max"    : float(np.max(vals)),
                "q10"    : float(np.quantile(vals, 0.10)),
                "q90"    : float(np.quantile(vals, 0.90)),
            })
        return rows


    def _print_stats_table(self, rows: list[dict], metric: str) -> None:
        """Print the stati  stic table on the terminal"""
        W = 100
        col = "{:>6} | {:>7} | {:>5} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}"
        print(f"\n{'='*W}")
        print(f"  Sample Efficiency — Statistics Table  [{metric}]")
        print(f"{'='*W}")
        print(col.format("k", "hours", "n", "mean", "median", "min", "max", "Q0.10", "Q0.90"))
        print("-" * W)
        for r in rows:
            print(col.format(
                r["k_shot"], f"{r['hours']:.1f}h", r["n"],
                f"{r['mean']:.5f}", f"{r['median']:.5f}",
                f"{r['min']:.5f}",  f"{r['max']:.5f}",
                f"{r['q10']:.5f}",  f"{r['q90']:.5f}",
            ))
        print("=" * W + "\n")


    def _save_stats_csv(self, rows: list[dict], metric: str) -> str:
        """Save the statistics table as CSV"""
        path = self.save_path + f"sample_efficiency_stats_{metric}.csv"
        fieldnames = ["k_shot", "hours", "n", "mean", "median", "min", "max", "q10", "q90"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {path}")
        return path
    

    # *****************************************************
    # ************** RUN'S COMPARISON *********************
    # *****************************************************

    # The code should look like this.
    # It is done to compare different supports
    # import pandas as pd

    # df1 = pd.read_csv("plots/job_A/sample_efficiency_stats_test_mae_final.csv")
    # df2 = pd.read_csv("plots/job_B/sample_efficiency_stats_test_mae_final.csv")

    # merged = df1.merge(df2, on="k_shot", suffixes=("_runA", "_runB"))
    # merged["delta_mean"] = merged["mean_runA"] - merged["mean_runB"]
    # print(merged[["k_shot", "hours_runA", "mean_runA", "mean_runB", "delta_mean"]])