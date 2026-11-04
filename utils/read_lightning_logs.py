#!/usr/bin/env python
"""
read_lightning_logs.py – Ultimate summary of a PyTorch Lightning TensorBoard run.

Features:
  • All available tags (scalars, histograms, images, audio, text)
  • Wide-format pivot (exploded table)
  • Per-tag statistics & percentiles
  • Meta train/val/test loss per epoch with visualization
  • Meta train/val/test loss per step (batch-level)
  • Convergence & overfitting analysis
  • Learning rate analysis
  • Gradient norm analysis
  • Hyperparameter loading
  • Histogram/weight analysis
  • Image & audio preview (if available)
  • Export to CSV/Parquet
  • Checkpoint analysis (if available)

Usage:
    utils/python read_lightning_logs.py <logdir> [options]
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from itertools import zip_longest
import re


try:
    from tbparse import SummaryReader
    import pandas as pd
except ImportError:
    sys.exit("Install required packages: pip install tbparse pandas numpy pyyaml")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hparams(logdir: Path) -> dict:
    """Load hyper‑parameters from hparams.yaml or hparams.json."""
    for name in ("hparams.yaml", "hparams.json"):
        hp_path = logdir / name
        if hp_path.exists():
            if name.endswith(".yaml"):
                import yaml
                with open(hp_path) as f:
                    return yaml.safe_load(f) or {}
            else:
                with open(hp_path) as f:
                    return json.load(f)
    return {}


def find_checkpoints(logdir: Path) -> list:
    """Find all checkpoint files in the logdir."""
    ckpt_dir = logdir / "checkpoints"
    if ckpt_dir.exists():
        return sorted(ckpt_dir.glob("*.ckpt"))
    return []


def pivot_to_wide(df) -> "pd.DataFrame":
    """Pivot scalar DataFrame to wide format (one column per tag)."""
    return (
        df.groupby(["step", "tag"])["value"]
        .mean()                  # aggregate duplicates instead of crashing
        .unstack("tag")
        .reset_index()
        .rename_axis(columns=None)
    )


def auto_rename(columns: list) -> dict:
    """Create a friendly renaming map for common Lightning tags."""
    rename = {}
    for col in columns:
        lower = col.lower()

        # masked first
        if "meta_train_zero_shot_loss_masked_epoch" in lower:
            rename[col] = "meta_train_zero_shot_loss_masked"
        elif "meta_train_zero_shot_loss_masked_step" in lower:
            rename[col] = "meta_train_zero_shot_loss_masked_step"
        elif "meta_train_support_loss_final_masked_epoch" in lower:
            rename[col] = "meta_train_support_loss_final_masked"
        elif "meta_train_support_loss_final_masked_step" in lower:
            rename[col] = "meta_train_support_loss_final_masked_step"
        elif "meta_train_loss_masked_epoch" in lower:
            rename[col] = "meta_train_loss_masked"
        elif "meta_train_loss_masked_step" in lower:
            rename[col] = "meta_train_loss_masked_step"
        elif "meta_val_loss_masked_epoch" in lower:
            rename[col] = "meta_val_loss_masked"
        elif "meta_val_loss_masked_step" in lower:
            rename[col] = "meta_val_loss_masked_step"
        elif "meta_test_loss_masked_epoch" in lower:
            rename[col] = "meta_test_loss_masked"
        elif "meta_test_loss_masked_step" in lower:
            rename[col] = "meta_test_loss_masked_step"
        elif "zero_shot_val_loss_masked_epoch" in lower:
            rename[col] = "zero_shot_val_loss_masked"
        elif "zero_shot_val_loss_masked_step" in lower:
            rename[col] = "zero_shot_val_loss_masked_step"
        elif "zero_shot_test_loss_masked_epoch" in lower:
            rename[col] = "zero_shot_test_loss_masked"
        elif "zero_shot_test_loss_masked_step" in lower:
            rename[col] = "zero_shot_test_loss_masked_step"

        # unmasked after
        elif "meta_train_zero_shot_loss_epoch" in lower:
            rename[col] = "meta_train_zero_shot_loss"
        elif "meta_train_zero_shot_loss_step" in lower:
            rename[col] = "meta_train_zero_shot_loss_step"
        elif "meta_train_support_loss_final_epoch" in lower:
            rename[col] = "meta_train_support_loss_final"
        elif "meta_train_support_loss_final_step" in lower:
            rename[col] = "meta_train_support_loss_final_step"
        elif "meta_train_loss_epoch" in lower:
            rename[col] = "meta_train_loss"
        elif "meta_train_loss_step" in lower:
            rename[col] = "meta_train_loss_step"
        elif "meta_val_loss_epoch" in lower:
            rename[col] = "meta_val_loss"
        elif "meta_val_loss_step" in lower:
            rename[col] = "meta_val_loss_step"
        elif "meta_test_loss_epoch" in lower:
            rename[col] = "meta_test_loss"
        elif "meta_test_loss_step" in lower:
            rename[col] = "meta_test_loss_step"
        elif "meta_loss_step" in lower:
            rename[col] = "meta_val_loss_step"
        elif "train_loss_epoch" in lower:
            rename[col] = "train_loss"
        elif "val_loss_epoch" in lower:
            rename[col] = "val_loss"
        elif "train_acc_epoch" in lower:
            rename[col] = "train_acc"
        elif "val_acc_epoch" in lower:
            rename[col] = "val_acc"
        elif lower.startswith("lr-"):
            rename[col] = "lr"
        elif "grad_norm" in lower:
            rename[col] = "grad_norm"
        elif "zero_shot_val_loss_epoch" in lower:
            rename[col] = "zero_shot_val_loss"
        elif "zero_shot_val_loss_step" in lower:
            rename[col] = "zero_shot_val_loss_step"
        elif "zero_shot_test_loss_epoch" in lower:
            rename[col] = "zero_shot_test_loss"
        elif "zero_shot_test_loss_step" in lower:
            rename[col] = "zero_shot_test_loss_step"

    return rename


def analyze_convergence(values: np.ndarray, steps: np.ndarray) -> dict:
    """Compute comprehensive convergence metrics."""
    if len(values) < 2:
        return {}

    best_idx = values.argmin()
    best_val = values[best_idx]
    best_step = steps[best_idx]
    final_val = values[-1]
    final_step = steps[-1]

    improvement = ((values[0] - best_val) / values[0] * 100) if values[0] > 0 else 0
    overfit_gap = final_val - best_val

    # Convergence (last 3 epochs)
    last_n = min(3, len(values))
    last_vals = values[-last_n:]
    variation = (last_vals.max() - last_vals.min()) / last_vals.mean() * 100 if last_vals.mean() > 0 else 0
    converged = variation < 5.0

    min_idx = values.argmin()
    early_stop_epoch = min_idx if min_idx < len(values) - 1 else None

    if len(values) > 1:
        reductions = np.diff(values)
        avg_reduction = reductions.mean()
        max_reduction = reductions.min()
    else:
        avg_reduction = max_reduction = 0

    return {
        "best_val": best_val,
        "best_step": best_step,
        "final_val": final_val,
        "final_step": final_step,
        "improvement_pct": improvement,
        "overfit_gap": overfit_gap,
        "overfit_gap_pct": (overfit_gap / best_val * 100) if best_val > 0 else 0,
        "converged": converged,
        "last_variation": variation,
        "early_stop_epoch": early_stop_epoch,
        "avg_reduction": avg_reduction,
        "max_reduction": max_reduction,
    }


def print_epoch_curve(phase_name: str, epoch_df) -> dict:
    """Print an epoch-by-epoch loss curve with bars, return convergence dict."""
    epoch_df = epoch_df.sort_values("step").reset_index(drop=True)

    print("\n" + "═" * 80)
    print(f"  📈 {phase_name} Loss per Epoch")
    print("═" * 80)
    print(f"  {'Epoch':>6}  {'Loss':>12}  {'Δ':>12}  {'Bar'}")
    print("  " + "-" * 55)

    values = epoch_df["value"].values
    steps = epoch_df["step"].values
    max_val, min_val = values.max(), values.min()

    for i, (step, val) in enumerate(zip(steps, values)):
        delta = val - values[i - 1] if i > 0 else 0.0
        bar_len = int(40 * (val - min_val) / (max_val - min_val + 1e-12)) if max_val > min_val else 0
        arrow = "↓" if delta < 0 else "↑" if delta > 0 else " "
        print(f"  {i:>6}  {val:>12.6f}  {arrow} {abs(delta):>10.6f}  {'█' * bar_len}")

    conv = analyze_convergence(values, steps)
    print(f"\n  🏆 Best  : {conv.get('best_val', values.min()):.6f} (step {conv.get('best_step', steps[values.argmin()])})")
    print(f"  🎯 Final : {conv.get('final_val', values[-1]):.6f}")
    print(f"  📉 Improvement: {conv.get('improvement_pct', 0):.1f}%")
    print(f"  📊 Converged  : {'✅ Yes' if conv.get('converged') else '❌ No'}")
    if conv.get('overfit_gap', 0) > 0:
        print(f"  ⚠️  Overfit gap: {conv['overfit_gap']:.6f} ({conv['overfit_gap_pct']:.1f}% above best)")
    return conv


def print_step_stats(phase_name: str, step_df):
    """Print batch-level statistics and percentiles for a step-tag."""
    vals = step_df["value"].values

    print("\n" + "═" * 80)
    print(f"  📊 {phase_name} Loss per Step (batch-level)")
    print("═" * 80)
    print(f"  Total steps     : {len(vals)}")
    print(f"  Mean            : {vals.mean():.6f}")
    print(f"  Std             : {vals.std():.6f}")
    print(f"  Median          : {np.median(vals):.6f}")

    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:02d} = {np.percentile(vals, p):.6f}")

    n_zeros = (vals == 0.0).sum()
    near_zeros = (vals < 1e-7).sum()
    if n_zeros > 0 or near_zeros > 0:
        print(f"\n  ⚠️  Exact zeros  : {n_zeros} / {len(vals)} ({100*n_zeros/len(vals):.1f}%)")
        print(f"  ⚠️  Near zeros   : {near_zeros} / {len(vals)} ({100*near_zeros/len(vals):.1f}%)")


def print_adaptation_gain(scalars: "pd.DataFrame"):
    """
    Compare zero-shot loss vs post-adaptation (meta) loss per epoch.
    Positive gain = adaptation helps; negative = adaptation hurts.
    """
    pairs = [
        ("Training", "meta_train_zero_shot_loss_epoch", "meta_train_loss_epoch"),
        ("Training Masked", "meta_train_zero_shot_loss_masked_epoch", "meta_train_loss_masked_epoch"),
        ("Validation", "zero_shot_val_loss_epoch", "meta_val_loss_epoch"),
        ("Test", "zero_shot_test_loss_epoch", "meta_test_loss_epoch"),
        ("Validation Masked", "zero_shot_val_loss_masked_epoch", "meta_val_loss_masked_epoch"),
        ("Test Masked", "zero_shot_test_loss_masked_epoch", "meta_test_loss_masked_epoch"),
    ]

    any_printed = False
    for split_name, zs_tag, meta_tag in pairs:
        zs_df   = scalars[scalars["tag"] == zs_tag].sort_values("step").reset_index(drop=True)
        meta_df = scalars[scalars["tag"] == meta_tag].sort_values("step").reset_index(drop=True)

        if zs_df.empty or meta_df.empty:
            continue
        any_printed = True

        # ── Align by step (inner join; fall back to positional) ──────────
        merged = pd.merge(
            zs_df[["step", "value"]].rename(columns={"value": "zero_shot"}),
            meta_df[["step", "value"]].rename(columns={"value": "meta"}),
            on="step", how="inner",
        ).sort_values("step").reset_index(drop=True)

        if merged.empty:                       # steps don't match → positional
            n = min(len(zs_df), len(meta_df))
            merged = pd.DataFrame({
                "step":      zs_df["step"].values[:n],
                "zero_shot": zs_df["value"].values[:n],
                "meta":      meta_df["value"].values[:n],
            })

        gains     = merged["zero_shot"] - merged["meta"]
        gains_pct = gains / merged["zero_shot"].replace(0, np.nan) * 100

        print("\n" + "═" * 80)
        print(f"  🚀 Adaptation Gain — {split_name}")
        print("═" * 80)
        print("  Definition: Gain = Zero-Shot Loss − Post-Adapt Loss")
        print("              Positive ↑ = adaptation helps  |  Negative ↓ = adaptation hurts")
        print()
        print(f"  {'Step':>6}  {'Zero-Shot':>12}  {'Post-Adapt':>12}  {'Gain':>12}  {'Gain %':>8}  {''}  ")
        print("  " + "─" * 58)

        for i, row in merged.iterrows():
            g     = row["zero_shot"] - row["meta"]
            g_pct = (g / row["zero_shot"] * 100) if row["zero_shot"] != 0 else 0.0
            flag  = "✅" if g > 0 else ("⚠️ " if g == 0 else "❌")
            print(
                f"  {int(row['step']):>6}  {row['zero_shot']:>12.6f}  "
                f"{row['meta']:>12.6f}  {g:>+12.6f}  {g_pct:>+7.1f}%  {flag}"
            )

        # ── Aggregate statistics ─────────────────────────────────────────
        print()
        n_pos = (gains > 0).sum()
        n_neg = (gains < 0).sum()
        n_neu = (gains == 0).sum()

        print(f"  Epochs w/ gain (adapt helps) : {n_pos}/{len(gains)}")
        print(f"  Epochs w/ loss (adapt hurts) : {n_neg}/{len(gains)}")
        print(f"  Epochs neutral               : {n_neu}/{len(gains)}")
        print()
        print(f"  Mean gain   : {gains.mean():>+.6f}  ({gains_pct.mean():>+.2f}%)")
        print(f"  Best gain   : {gains.max():>+.6f}  "
              f"(step {merged.loc[gains.idxmax(), 'step']})")
        print(f"  Worst gain  : {gains.min():>+.6f}  "
              f"(step {merged.loc[gains.idxmin(), 'step']})")
        print(f"  Final gain  : {gains.iloc[-1]:>+.6f}  ({gains_pct.iloc[-1]:>+.2f}%)")

        # ── Trend: is gain growing over training? ────────────────────────
        if len(gains) >= 4:
            mid         = len(gains) // 2
            first_half  = gains.iloc[:mid].mean()
            second_half = gains.iloc[mid:].mean()
            trend       = second_half - first_half
            print()
            if trend > 1e-6:
                print(f"  📈 Gain trend: +{trend:.6f} → adaptation improving over training  ✅")
            elif trend < -1e-6:
                print(f"  📉 Gain trend:  {trend:.6f} → adaptation weakening over training  ⚠️")
            else:
                print(f"  ➡️  Gain trend: ~0 → stable throughout training")

        # ── Verdict ──────────────────────────────────────────────────────
        mean_g = gains.mean()
        print()
        if mean_g > 0.05:
            print("  🏆 Verdict: Strong adaptation — meta-learning is working well.")
        elif mean_g > 1e-4:
            print("  ✅ Verdict: Mild but consistent adaptation gain.")
        elif mean_g >= 0:
            print("  ⚠️  Verdict: Near-zero gain — inner loop may need more steps or higher LR.")
        else:
            print("  ❌ Verdict: Negative gain — post-adapt loss > zero-shot. Debug inner loop.")

    if not any_printed:
        print("\n  ⚠️  Adaptation Gain: no zero_shot_*_loss_epoch tags found — skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def summarise(logdir: str, export_csv: bool = False, export_parquet: bool = False,
              show_images: bool = False, show_histograms: bool = True):
    logdir = Path(logdir)
    reader = SummaryReader(logdir)
    scalars = reader.scalars

    if scalars.empty:
        print("❌ No scalar data found.")
        return

    # ── Load hyper‑parameters & checkpoints ───────────────────────────────
    hparams = load_hparams(logdir)
    checkpoints = find_checkpoints(logdir)

    # ── Header ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print(f"  📊 TensorBoard Summary: {logdir.name}")
    print(f"  📅 Analysis run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    # ── Files in directory ────────────────────────────────────────────────
    print(f"\n📁 Files in {logdir.name}:")
    all_files = sorted(logdir.rglob("*"))
    file_types = {}
    for f in all_files:
        if f.is_file():
            ext = f.suffix or "no_ext"
            file_types[ext] = file_types.get(ext, 0) + 1
    for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
        print(f"   • {ext}: {count} files")

    # ── Checkpoints ────────────────────────────────────────────────────────
    if checkpoints:
        print(f"\n🏋️  Checkpoints found: {len(checkpoints)}")
        for ckpt in checkpoints[:5]:
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"   • {ckpt.name} ({size_mb:.1f} MB)")
        if len(checkpoints) > 5:
            print(f"   ... and {len(checkpoints) - 5} more")
    else:
        print("\n🏋️  No checkpoints found")

    # ── Available tags ───────────────────────────────────────────────────
    tags = sorted(scalars["tag"].unique())
    print(f"\n📌 Available scalar tags ({len(tags)}):")
    for t in tags:
        print(f"   • {t}")

    # ── Wide‑format table (the "explode") ────────────────────────────────
    wide_df = pivot_to_wide(scalars)
    rename_map = auto_rename(list(wide_df.columns))
    if rename_map:
        wide_df = wide_df.rename(columns=rename_map)

    preview_cols = ["step"] + [c for c in wide_df.columns if c != "step"][:5]
    print("\n" + "─" * 80)
    print("  📈 Wide‑format preview (first 6 rows, max 5 metrics):")
    print("─" * 80)
    print(wide_df[preview_cols].head(6).to_string(index=False))
    print(f"\n  ℹ️  Wide table shape: {wide_df.shape[0]} steps × {wide_df.shape[1]} columns")

    # ── Export option ────────────────────────────────────────────────────
    if export_csv:
        out_csv = logdir.parent / f"{logdir.name}_exploded.csv"
        wide_df.to_csv(out_csv, index=False)
        print(f"\n  💾 Exported wide table to CSV: {out_csv}")

    if export_parquet:
        out_parquet = logdir.parent / f"{logdir.name}_exploded.parquet"
        wide_df.to_parquet(out_parquet, index=False)
        print(f"\n  💾 Exported wide table to Parquet: {out_parquet}")

    # ── Per-tag statistics ────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  📉 Per‑tag statistics:")
    print("─" * 80)
    print(f"{'Tag':<35} {'Count':>6} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    for tag in tags:
        vals = scalars[scalars["tag"] == tag]["value"].values
        print(
            f"{tag:<35} {len(vals):>6} "
            f"{vals.min():>12.6f} {vals.max():>12.6f} "
            f"{vals.mean():>12.6f} {vals.std():>12.6f}"
        )

    # ── Learning Rate Analysis ───────────────────────────────────────────
    lr_tags = [t for t in tags if t.lower().startswith("lr-")]
    if lr_tags:
        print("\n" + "═" * 80)
        print("  📉 Learning Rate Analysis")
        print("═" * 80)
        for lr_tag in lr_tags:
            lr_vals = scalars[scalars["tag"] == lr_tag]["value"].values
            print(f"  {lr_tag}:")
            print(f"    Start: {lr_vals[0]:.2e}, End: {lr_vals[-1]:.2e}, "
                  f"Change: {((lr_vals[-1] - lr_vals[0]) / lr_vals[0] * 100):.1f}%")
            if len(lr_vals) > 1:
                reductions = np.where(np.diff(lr_vals) < 0)[0]
                if len(reductions) > 0:
                    print(f"    LR reductions at steps: {reductions[:5]}...")

    # ── Gradient Norm Analysis ───────────────────────────────────────────
    grad_tags = [t for t in tags if "grad/" in t.lower() or "grad_norm" in t.lower()]
    if grad_tags:
        print("\n" + "═" * 80)
        print("  📉 Gradient Analysis")
        print("═" * 80)
        for grad_tag in grad_tags:
            grad_vals = scalars[scalars["tag"] == grad_tag]["value"].values
            print(f"  {grad_tag}:")
            print(f"    Mean: {grad_vals.mean():.6f}, Max: {grad_vals.max():.6f}, "
                f"Min: {grad_vals.min():.6f}")
            exploding = (grad_vals > 10.0).sum()
            if exploding > 0:
                print(f"    ⚠️  Large values (>10): {exploding} times "
                    f"({100*exploding/len(grad_vals):.1f}%)")

    # ── Weight Norm Analysis ─────────────────────────────────────────────
    weight_tags = [t for t in tags if t.lower().startswith("weight/")]
    if weight_tags:
        print("\n" + "═" * 80)
        print("  ⚖️  Weight Norm Analysis")
        print("═" * 80)
        for weight_tag in weight_tags:
            vals = scalars[scalars["tag"] == weight_tag]["value"].values
            print(f"  {weight_tag}:")
            print(f"    Mean: {vals.mean():.6f}, Max: {vals.max():.6f}, "
                f"Min: {vals.min():.6f}")

    # ── Grad/Weight Ratio Analysis ───────────────────────────────────────
    ratio_tags = [t for t in tags if "grad_ratio/" in t.lower()]
    if ratio_tags:
        print("\n" + "═" * 80)
        print("  📐 Grad/Weight Ratio Analysis")
        print("═" * 80)
        for ratio_tag in ratio_tags:
            vals = scalars[scalars["tag"] == ratio_tag]["value"].values
            print(f"  {ratio_tag}:")
            print(f"    Mean: {vals.mean():.6e}, Max: {vals.max():.6e}, "
                f"Min: {vals.min():.6e}")

    # ── Meta loss per epoch (train / val / test) ─────────────────────────
    epoch_phase_tags = {
        "Meta-Training": "meta_train_loss_epoch",
        "Meta-Training Masked": "meta_train_loss_masked_epoch",
        "Meta-Training Zero-Shot": "meta_train_zero_shot_loss_epoch",
        "Meta-Training Zero-Shot Masked": "meta_train_zero_shot_loss_masked_epoch",
        "Meta-Training Support Final": "meta_train_support_loss_final_epoch",
        "Meta-Training Support Final Masked": "meta_train_support_loss_final_masked_epoch",

        "Meta-Validation": "meta_val_loss_epoch",
        "Meta-Testing": "meta_test_loss_epoch",
        "Meta (legacy)": "meta_loss_epoch",

        "Zero-Shot Val": "zero_shot_val_loss_epoch",
        "Zero-Shot Test": "zero_shot_test_loss_epoch",
        "Zero-Shot Val Masked": "zero_shot_val_loss_masked_epoch",
        "Zero-Shot Test Masked": "zero_shot_test_loss_masked_epoch",
    }
    for phase_name, tag in epoch_phase_tags.items():
        epoch_df = scalars[scalars["tag"] == tag].copy()
        if epoch_df.empty:
            continue
        print_epoch_curve(phase_name, epoch_df)

    # ── Meta loss per step (train / val / test) ──────────────────────────
    step_phase_tags = {
        "Meta-Training": "meta_train_loss_step",
        "Meta-Training Masked": "meta_train_loss_masked_step",
        "Meta-Training Zero-Shot": "meta_train_zero_shot_loss_step",
        "Meta-Training Zero-Shot Masked": "meta_train_zero_shot_loss_masked_step",
        "Meta-Training Support Final": "meta_train_support_loss_final_step",
        "Meta-Training Support Final Masked": "meta_train_support_loss_final_masked_step",

        "Meta-Validation": "meta_val_loss_step",
        "Meta-Testing": "meta_test_loss_step",
        "Meta (legacy)": "meta_loss_step",

        "Zero-Shot Val": "zero_shot_val_loss_step",
        "Zero-Shot Test": "zero_shot_test_loss_step",
        "Zero-Shot Val Masked": "zero_shot_val_loss_masked_step",
        "Zero-Shot Test Masked": "zero_shot_test_loss_masked_step",
        "Meta-Validation Masked": "meta_val_loss_masked_step",
        "Meta-Testing Masked": "meta_test_loss_masked_step",
    }
    for phase_name, tag in step_phase_tags.items():
        step_df = scalars[scalars["tag"] == tag].copy()
        if step_df.empty:
            continue
        print_step_stats(phase_name, step_df)

    # ── Meta Train vs Meta Val Gap (overfitting) ─────────────────────────
    train_df = scalars[scalars["tag"] == "meta_train_loss_epoch"].sort_values("step")
    val_df = scalars[scalars["tag"] == "meta_val_loss_epoch"].sort_values("step")

    if not train_df.empty and not val_df.empty:
        print("\n" + "═" * 80)
        print("  📈 Meta Train vs Meta Val Gap (overfitting)")
        print("═" * 80)

        train_vals = train_df["value"].values
        val_vals = val_df["value"].values
        n = min(len(train_vals), len(val_vals))   # align on shorter curve

        if n == 0:
            print("  ⚠️  One of the curves is empty — skipping.")
        else:
            train_vals = train_vals[:n]
            val_vals = val_vals[:n]
            gaps = val_vals - train_vals

            print(f"  {'Epoch':>6}  {'Train':>12}  {'Val':>12}  {'Gap':>12}")
            print("  " + "-" * 45)
            for i in range(min(15, n)):
                print(f"  {i:>6}  {train_vals[i]:>12.6f}  {val_vals[i]:>12.6f}  {gaps[i]:>12.6f}")
            if n > 15:
                print("  ...")

            print(f"\n  Avg gap  : {gaps.mean():.6f} (std: {gaps.std():.6f})")
            print(f"  Final gap: {gaps[-1]:.6f}")

            if gaps[-1] > gaps[0] and gaps[-1] > 0:
                print("  ⚠️  Val gap widening over training → possible overfitting.")
            elif gaps[-1] <= 0:
                print("  ✅ Val ≤ Train at end → no overfitting signal.")
            else:
                print("  ✅ Gap stable / shrinking → healthy generalization.")

    # ── Generic Train vs Val Gap (non-meta runs) ─────────────────────────
    train_loss_tags = [t for t in tags if t == "train_loss_epoch"]
    val_loss_tags = [t for t in tags if t == "val_loss_epoch"]
    if train_loss_tags and val_loss_tags:
        print("\n" + "═" * 80)
        print("  📈 Training vs Validation Gap")
        print("═" * 80)
        train_vals = scalars[scalars["tag"] == train_loss_tags[0]]["value"].values
        val_vals = scalars[scalars["tag"] == val_loss_tags[0]]["value"].values

        if len(train_vals) == len(val_vals):
            gaps = val_vals - train_vals
            print(f"  {'Epoch':>6}  {'Train':>12}  {'Val':>12}  {'Gap':>12}")
            print("  " + "-" * 45)
            for i in range(min(10, len(train_vals))):
                print(f"  {i:>6}  {train_vals[i]:>12.6f}  {val_vals[i]:>12.6f}  {gaps[i]:>12.6f}")
            if len(train_vals) > 10:
                print("  ...")

            print(f"\n  Avg gap: {gaps.mean():.6f} (std: {gaps.std():.6f})")
            print(f"  Final gap: {gaps[-1]:.6f}")

    print_adaptation_gain(scalars)

    # ── Hyper‑parameters (if any) ────────────────────────────────────────
    if hparams:
        print("\n" + "═" * 80)
        print("  ⚙️  Hyper‑parameters")
        print("═" * 80)
        keys = list(hparams.keys())
        mid = (len(keys) + 1) // 2
        col1, col2 = keys[:mid], keys[mid:]
        max_key_len = max(len(str(k)) for k in keys) if keys else 0
        # for k1, k2 in zip(col1, col2):
        #     v1 = str(hparams[k1])[:30]
        #     v2 = str(hparams[k2])[:30] if k2 else ""
        #     print(f"  {k1:<{max_key_len}} : {v1:<30}    {k2:<{max_key_len}} : {v2}")
        
        for k1, k2 in zip_longest(col1, col2, fillvalue=None):
            v1 = str(hparams[k1])[:30] if k1 is not None else ""
            v2 = str(hparams[k2])[:30] if k2 is not None else ""
            k1s = k1 if k1 is not None else ""
            k2s = k2 if k2 is not None else ""
            print(f"  {k1s:<{max_key_len}} : {v1:<30}    {k2s:<{max_key_len}} : {v2}")
        
        if len(col1) > len(col2):
            k1 = col1[-1]
            print(f"  {k1:<{max_key_len}} : {str(hparams[k1])[:30]}")

    # ── Histogram summary ───────────────────────────────────────────────
    if show_histograms:
        try:
            hist_df = reader.histograms
            if not hist_df.empty:
                print("\n" + "═" * 80)
                print("  📊 Histogram summary (weight distributions)")
                print("═" * 80)
                hist_tags = hist_df["tag"].unique()
                print(f"  Found {len(hist_tags)} histogram tags, showing first 8:")
                for tag in sorted(hist_tags)[:8]:
                    hvals = hist_df[hist_df["tag"] == tag]["value"]
                    last_hist = hvals.iloc[-1] if len(hvals) > 0 else []
                    if isinstance(last_hist, list) and len(last_hist) > 0:
                        bins = np.array(last_hist)[:, 0]
                        counts = np.array(last_hist)[:, 1]
                        mean_w = np.average(bins, weights=counts) if counts.sum() > 0 else 0.0
                        std_w = np.sqrt(np.average((bins - mean_w) ** 2, weights=counts)) if counts.sum() > 0 else 0.0
                        print(f"    • {tag:<45} mean={mean_w:>10.4f}  std={std_w:>10.4f}")
                if len(hist_tags) > 8:
                    print(f"    ... and {len(hist_tags) - 8} more.")
        except Exception as e:
            print(f"\n  ⚠️  Could not read histograms: {e}")

    # ── Image preview ───────────────────────────────────────────────────
    if show_images:
        try:
            img_df = reader.images
            if not img_df.empty:
                print("\n" + "═" * 80)
                print("  🖼️  Image summary")
                print("═" * 80)
                img_tags = img_df["tag"].unique()
                print(f"  Found {len(img_tags)} image tags:")
                for tag in sorted(img_tags)[:5]:
                    count = len(img_df[img_df["tag"] == tag])
                    print(f"    • {tag}: {count} images")
        except Exception as e:
            print(f"\n  ⚠️  Could not read images: {e}")

    # ── Footer ───────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ✅ Done!")
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultimate summary of a PyTorch Lightning TensorBoard run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("logdir", nargs="?", default="lightning_logs/version_216704",
                        help="Path to the lightning_logs version folder")
    parser.add_argument("--export-csv", action="store_true", help="Export exploded wide table to CSV")
    parser.add_argument("--export-parquet", action="store_true", help="Export exploded wide table to Parquet")
    parser.add_argument("--show-images", action="store_true", help="Show image summary")
    parser.add_argument("--no-histograms", action="store_true", help="Skip histogram summary")
    args = parser.parse_args()

    if not Path(args.logdir).exists():
        sys.exit(f"❌ Path not found: {args.logdir}")

    summarise(args.logdir,
              export_csv=args.export_csv,
              export_parquet=args.export_parquet,
              show_images=args.show_images,
              show_histograms=not args.no_histograms)
