# ks_analysis.py
"""
KS-based distribution analysis for Few-Shot Solar Forecasting (Test Set only).

Test A : y_support  vs y_query   → Covariate Shift
Test B : pred_median vs y_query  → Calibration
Coverage: % y_query inside [q_low, q_high]

Only masked values (y > threshold) are used.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Main Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class KSDistributionAnalyzer:
    """
    Args:
        pred_cache      : List[(preds [B,T,Q], y_query [B,T])]
        support_cache   : List[(X_support [*], y_support [B,T])]
        plant_id_cache  : List[{"plant_id": str, "rated_power_kW": float, ...}]
        threshold       : values <= threshold are excluded (zero masking)
        quantile_idx    : (idx_low, idx_median, idx_high) in preds last dim
        output_dir      : directory for CSVs and plots
    """

    def __init__(
        self,
        pred_cache: List,
        support_cache: List,
        plant_id_cache: List[Dict],
        threshold: float = 0.0,
        quantile_idx: Tuple[int, int, int] = (0, 1, 2),
        output_dir: Optional[Path] = None,
    ):
        assert len(pred_cache) == len(support_cache) == len(plant_id_cache), (
            f"Cache length mismatch: pred={len(pred_cache)}, "
            f"sup={len(support_cache)}, plant={len(plant_id_cache)}"
        )
        self.pred_cache      = pred_cache
        self.support_cache   = support_cache
        self.plant_id_cache  = plant_id_cache
        self.threshold       = threshold
        self.q_lo, self.q_md, self.q_hi = quantile_idx
        self.output_dir      = Path(output_dir or "ks_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full analysis. Returns (per_plant_df, aggregate_df)."""
        log.info("▶ KS analysis started — %d entries in cache", len(self.pred_cache))

        plant_data   = self._group_by_plant()
        per_plant_df = pd.DataFrame([
            self._analyze_plant(pid, d) for pid, d in plant_data.items()
        ])
        aggregate_df = self._compute_aggregate(per_plant_df)

        # Save
        per_plant_df.to_csv(self.output_dir / "ks_per_plant.csv",  index=False)
        aggregate_df.to_csv(self.output_dir / "ks_aggregate.csv")

        # Plots
        self._plot_cdfs(plant_data)
        self._plot_summary(per_plant_df)

        self._log_summary(per_plant_df, aggregate_df)
        log.info("✔ KS analysis complete → %s", self.output_dir)
        return per_plant_df, aggregate_df

    # ── Data Grouping ─────────────────────────────────────────────────────────

    def _group_by_plant(self) -> Dict[str, Dict]:
        """Merge all caches → {plant_id: {y_support, y_query, pred_low/med/high}}"""
        plant_data: Dict[str, Dict] = {}

        for meta, pred_entry, sup_entry in zip(
            self.plant_id_cache, self.pred_cache, self.support_cache
        ):
            pid   = str(meta.get("plant_id", "unknown"))
            preds, y_query   = self._extract_preds(pred_entry)   # [N,3], [N]
            y_support        = self._extract_support(sup_entry)  # [M]

            if pid not in plant_data:
                plant_data[pid] = {
                    "meta": meta,
                    "y_support": [], "y_query": [],
                    "pred_lo": [], "pred_md": [], "pred_hi": [],
                }

            plant_data[pid]["y_support"].append(y_support)
            plant_data[pid]["y_query"].append(y_query)
            plant_data[pid]["pred_lo"].append(preds[:, self.q_lo])
            plant_data[pid]["pred_md"].append(preds[:, self.q_md])
            plant_data[pid]["pred_hi"].append(preds[:, self.q_hi])

        # Flatten lists → single arrays
        for pid in plant_data:
            for k in ["y_support", "y_query", "pred_lo", "pred_md", "pred_hi"]:
                plant_data[pid][k] = np.concatenate(plant_data[pid][k])

        return plant_data

    # ── Extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    def _extract_support(self, entry) -> np.ndarray:
        _, y = entry   # ← X_train wird mit _ weggeworfen, nur y zählt
        y = self._to_np(y)
        return y.ravel()

    def _extract_preds(self, entry) -> Tuple[np.ndarray, np.ndarray]:
        """
        From pred_cache entry: (preds, y_query)
        preds  shape [B, T, Q] or [N, Q]  →  [N, Q]
        y_query shape [B, T]  or [N]       →  [N]
        """
        preds, y = entry
        preds = self._to_np(preds)
        y     = self._to_np(y)
        # ── DEBUG ──────────────────────────────
        log.debug("y_query: shape=%s  min=%.4f  max=%.4f  nan=%d  zeros=%d",
                y.shape, np.nanmin(y), np.nanmax(y),
                np.isnan(y).sum(), (y == 0.0).sum())
        # ───────────────────────────────────────

        if preds.ndim == 3:            # [B, T, Q]
            B, T, Q = preds.shape
            preds = preds.reshape(-1, Q)
            y     = y.reshape(-1)
        elif preds.ndim == 2 and preds.shape[-1] != 3:
            # Might be [B, T] — shouldn't happen, guard anyway
            preds = np.stack([preds.ravel()] * 3, axis=-1)
            y     = y.ravel()
        else:                          # [N, Q]
            preds = preds.reshape(-1, preds.shape[-1])
            y     = y.ravel()

        # ── Guard: zu wenige Quantile ─────────────────────────────────────
        needed = max(self.q_lo, self.q_md, self.q_hi) + 1
        if preds.shape[1] < needed:
            log.warning(
                "_extract_preds: preds has only %d quantile column(s), "
                "but quantile_idx=%s requires %d. Replicating columns.",
                preds.shape[1], (self.q_lo, self.q_md, self.q_hi), needed,
            )
            preds = np.tile(preds, (1, needed))[:, :needed]
        # ──────────────────────────────────────────────────────────────────


        return preds, y

    def _extract_support(self, entry) -> np.ndarray:
        """From support_cache entry: (X_support, y_support) → y_support flat [M]"""
        _, y = entry
        y = self._to_np(y)
        return y.ravel()

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _mask(self, arr: np.ndarray) -> np.ndarray:
        return arr[arr > self.threshold]

    def _analyze_plant(self, plant_id: str, data: Dict) -> Dict:
        y_sup = self._mask(data["y_support"])
        y_qry = self._mask(data["y_query"])
        p_med = self._mask(data["pred_md"])

        # For coverage: align pred_lo/hi with y_query mask
        mask_q  = data["y_query"] > self.threshold
        y_aligned  = data["y_query"][mask_q]
        plo_aligned = data["pred_lo"][mask_q]
        phi_aligned = data["pred_hi"][mask_q]

        base = {
            "plant_id":           plant_id,
            "rated_power_kW":     data["meta"].get("rated_power_kW", np.nan),
            "n_support_masked":   len(y_sup),
            "n_query_masked":     len(y_qry),
        }

        if len(y_sup) < 10 or len(y_qry) < 10:
            log.warning("Plant %s: too few samples after masking (sup=%d, qry=%d)",
                        plant_id, len(y_sup), len(y_qry))
            return {**base,
                    "ks_A_stat": np.nan, "ks_A_pval": np.nan, "ks_A_sig": False,
                    "ks_B_stat": np.nan, "ks_B_pval": np.nan, "ks_B_sig": False,
                    "coverage_pct": np.nan}

        ks_A = stats.ks_2samp(y_sup, y_qry)
        ks_B = stats.ks_2samp(p_med, y_qry)

        inside       = (y_aligned >= plo_aligned) & (y_aligned <= phi_aligned)
        coverage_pct = 100.0 * inside.mean() if len(inside) > 0 else np.nan

        return {
            **base,
            # Test A
            "ks_A_stat":     round(float(ks_A.statistic), 4),
            "ks_A_pval":     round(float(ks_A.pvalue),    6),
            "ks_A_sig":      bool(ks_A.pvalue < 0.05),
            # Test B
            "ks_B_stat":     round(float(ks_B.statistic), 4),
            "ks_B_pval":     round(float(ks_B.pvalue),    6),
            "ks_B_sig":      bool(ks_B.pvalue < 0.05),
            # Coverage
            "coverage_pct":  round(float(coverage_pct), 2),
        }

    def _compute_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = ["ks_A_stat", "ks_A_pval", "ks_B_stat", "ks_B_pval", "coverage_pct"]
        agg = df[numeric].agg(["mean", "median", "std", "min", "max"]).round(4)

        # Append significance summary as extra rows
        sig = pd.DataFrame({
            "ks_A_stat":    [df["ks_A_sig"].mean() * 100],
            "ks_A_pval":    [np.nan],
            "ks_B_stat":    [df["ks_B_sig"].mean() * 100],
            "ks_B_pval":    [np.nan],
            "coverage_pct": [np.nan],
        }, index=["pct_significant"])

        return pd.concat([agg, sig])

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot_cdfs(self, plant_data: Dict):
        """One CDF figure per plant (2 panels: Test A and Test B)."""
        for plant_id, data in plant_data.items():
            y_sup = self._mask(data["y_support"])
            y_qry = self._mask(data["y_query"])
            p_med = self._mask(data["pred_md"])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"KS Distribution Analysis — {plant_id}  "
                f"(threshold={self.threshold})",
                fontsize=12, fontweight="bold"
            )

            self._draw_cdf(
                axes[0],
                [(y_sup, "Support", "#2196F3"),
                 (y_qry, "Query",   "#FF5722")],
                title="Test A — Support vs Query  (Covariate Shift)",
            )
            self._draw_cdf(
                axes[1],
                [(p_med, "Pred Median", "#4CAF50"),
                 (y_qry, "Query",       "#FF5722")],
                title="Test B — Pred Median vs Query  (Calibration)",
            )

            plt.tight_layout()
            fig.savefig(
                self.output_dir / f"cdf_{plant_id}.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

    @staticmethod
    def _draw_cdf(ax, series: List[Tuple], title: str):
        for values, label, color in series:
            sv = np.sort(values)
            p  = np.arange(1, len(sv) + 1) / len(sv)
            ax.plot(sv, p, label=f"{label}  (n={len(values):,})",
                    color=color, linewidth=1.8)
        ax.set_xlabel("Normalised Power Output")
        ax.set_ylabel("CDF")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_summary(self, df: pd.DataFrame):
        """Bar chart summary: KS-A, KS-B statistics + coverage per plant."""
        valid = df.dropna(subset=["ks_A_stat"]).sort_values("ks_A_stat", ascending=False)
        n     = len(valid)
        x     = np.arange(n)
        labels = valid["plant_id"].astype(str).tolist()
        w      = max(12, n * 0.55)

        fig, axes = plt.subplots(3, 1, figsize=(w, 13), sharex=False)
        fig.suptitle("KS Analysis — All Plants Summary", fontsize=13, fontweight="bold")

        def _bar(ax, values, colors, title, ylabel, hline=None, hline_label=None):
            ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            if hline is not None:
                ax.axhline(hline, ls="--", color="gray", alpha=0.7, label=hline_label)
                ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        sig_colors_A = ["#FF5722" if s else "#2196F3" for s in valid["ks_A_sig"]]
        sig_colors_B = ["#FF5722" if s else "#4CAF50" for s in valid["ks_B_sig"]]
        cov_colors   = ["#9C27B0"] * n

        _bar(axes[0], valid["ks_A_stat"], sig_colors_A,
             "Test A — KS Statistic: Support vs Query  (red = p < 0.05)",
             "KS Statistic D", hline=0.1, hline_label="D=0.1 ref")

        _bar(axes[1], valid["ks_B_stat"], sig_colors_B,
             "Test B — KS Statistic: Pred Median vs Query  (red = p < 0.05)",
             "KS Statistic D", hline=0.1)

        _bar(axes[2], valid["coverage_pct"], cov_colors,
             "Quantile Coverage  (% of y_query inside [q_low, q_high])",
             "Coverage (%)", hline=80.0, hline_label="80% reference")
        axes[2].set_ylim(0, 105)

        plt.tight_layout()
        fig.savefig(self.output_dir / "ks_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Logging ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(per_plant_df: pd.DataFrame, agg_df: pd.DataFrame):
        df = per_plant_df.dropna(subset=["ks_A_stat"])
        n  = len(df)
        log.info("──────── KS Summary (%d plants) ────────", n)
        log.info("Test A (Support vs Query):  D̄=%.3f  |  %.0f%% significant",
                 df["ks_A_stat"].mean(), df["ks_A_sig"].mean() * 100)
        log.info("Test B (Pred vs Query):     D̄=%.3f  |  %.0f%% significant",
                 df["ks_B_stat"].mean(), df["ks_B_sig"].mean() * 100)
        log.info("Coverage (q_lo–q_hi):       mean=%.1f%%  median=%.1f%%",
                 df["coverage_pct"].mean(), df["coverage_pct"].median())
        log.info("────────────────────────────────────────")