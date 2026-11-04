#!/usr/bin/env python3
"""
Cherry-pick visual analysis — 3×3 plot grid.
Usage: python cherry_pick_plots.py path/to/per_plant_test_results.csv [TOP_N]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "per_plant_test_results.csv"
TOP_N    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT_PATH = Path(CSV_PATH).parent / "cherry_pick_plots.png"

# ── Palette ───────────────────────────────────────────────────────────────────
C_ZERO  = "#E07B54"   # orange-red  → zero-shot
C_ADAPT = "#4C9BE8"   # steel blue  → adapted
C_GAIN  = "#5DBB63"   # green       → positive gain annotation
C_HURT  = "#E05454"   # red         → negative gain
C_PWR   = "#9B7FC7"   # purple      → rated power
C_BG    = "#F8F9FA"
C_GRID  = "#DEDEDE"

# ── Load & enrich ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["zero_shot_MAE", "adapted_MAE", "gain_abs", "gain_pct"])
df["nMAE_zero_shot"] = df["zero_shot_MAE"] / df["rated_power_kW"].replace(0, np.nan)
df["nMAE_adapted"]   = df["adapted_MAE"]   / df["rated_power_kW"].replace(0, np.nan)

def short_id(pid, n=8):
    s = str(pid)
    return f"…{s[-n:]}" if len(s) > n else s

# ── Shared drawing helpers ────────────────────────────────────────────────────
def _style(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=7)
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

def _annotate_gain(ax, x_pos, y_pos, gain_pct):
    color = C_GAIN if gain_pct >= 0 else C_HURT
    sign  = "+" if gain_pct >= 0 else ""
    ax.annotate(f"{sign}{gain_pct:.1f}%",
                xy=(x_pos, y_pos), fontsize=7.5,
                ha="center", va="bottom",
                color=color, fontweight="bold")

def _empty(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=7)
    ax.set_facecolor(C_BG)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, "No data for\nthis criterion",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray", style="italic")


def plot_grouped(ax, subset, title,
                 col_a=C_ZERO, col_b=C_ADAPT,
                 ycol_a="zero_shot_MAE", ycol_b="adapted_MAE",
                 label_a="Zero-Shot", label_b="Adapted",
                 ylabel="MAE"):
    """Grouped bar: two metrics side-by-side + gain % annotation."""
    sub = subset.reset_index(drop=True)
    if sub.empty:
        _empty(ax, title); return

    n, w = len(sub), 0.35
    x = np.arange(n)

    ax.bar(x - w/2, sub[ycol_a], w, color=col_a, label=label_a,
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, sub[ycol_b], w, color=col_b, label=label_b,
           edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short_id(p) for p in sub["plant_id"]],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, framealpha=0.7)
    _style(ax, title)

    ymax = max(sub[ycol_a].max(), sub[ycol_b].max())
    for i, row in sub.iterrows():
        y_top = max(row[ycol_a], row[ycol_b]) + ymax * 0.03
        _annotate_gain(ax, x[i], y_top, row["gain_pct"])


def plot_power_context(ax, subset, title):
    """MAE bars + rated_power as secondary axis (purple)."""
    sub = subset.reset_index(drop=True)
    if sub.empty:
        _empty(ax, title); return

    n, w = len(sub), 0.28
    x = np.arange(n)

    ax.bar(x - w, sub["zero_shot_MAE"],  w, color=C_ZERO,  label="Zero-Shot",
           edgecolor="white", zorder=3)
    ax.bar(x,     sub["adapted_MAE"],    w, color=C_ADAPT, label="Adapted",
           edgecolor="white", zorder=3)

    ax2 = ax.twinx()
    ax2.bar(x + w, sub["rated_power_kW"], w, color=C_PWR, alpha=0.45,
            label="Rated Power (kW)", edgecolor="white", zorder=2)
    ax2.set_ylabel("Rated Power (kW)", fontsize=8, color=C_PWR)
    ax2.tick_params(axis="y", labelcolor=C_PWR, labelsize=7)
    ax2.spines[["top"]].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([short_id(p) for p in sub["plant_id"]],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    handles = [mpatches.Patch(color=C_ZERO,  label="Zero-Shot"),
               mpatches.Patch(color=C_ADAPT, label="Adapted"),
               mpatches.Patch(color=C_PWR,   alpha=0.6, label="Rated Power")]
    ax.legend(handles=handles, fontsize=7, framealpha=0.7)
    _style(ax, title)

    ymax = max(sub["zero_shot_MAE"].max(), sub["adapted_MAE"].max())
    for i, row in sub.iterrows():
        y_top = max(row["zero_shot_MAE"], row["adapted_MAE"]) + ymax * 0.03
        _annotate_gain(ax, x[i], y_top, row["gain_pct"])


# ── Panel definitions ─────────────────────────────────────────────────────────
panels = [
    dict(title="① Hardest Start\n(Max Zero-Shot MAE)",
         fn=plot_grouped,
         subset=df.nlargest(TOP_N, "zero_shot_MAE")),

    dict(title="② Best Relative Gain\n(Max gain_pct)",
         fn=plot_grouped,
         subset=df[df.gain_pct > 0].nlargest(TOP_N, "gain_pct")),

    dict(title="③ Best Absolute Gain\n(Max gain_abs)",
         fn=plot_grouped,
         subset=df[df.gain_abs > 0].nlargest(TOP_N, "gain_abs")),

    dict(title="④ Easiest Start\n(Min Zero-Shot MAE)",
         fn=plot_grouped,
         subset=df.nsmallest(TOP_N, "zero_shot_MAE")),

    dict(title="⑤ Best Final Result\n(Min Adapted MAE)",
         fn=plot_grouped,
         subset=df.nsmallest(TOP_N, "adapted_MAE")),

    dict(title="⑥ Best Scale-Free Perf.\n(Min nMAE_adapted = MAE/kW)",
         fn=plot_grouped,
         subset=df.nsmallest(TOP_N, "nMAE_adapted"),
         kwargs=dict(ycol_a="nMAE_zero_shot", ycol_b="nMAE_adapted",
                     ylabel="MAE / rated_power_kW")),

    dict(title="⑦ Adaptation HURT\n(Most Negative gain_pct)",
         fn=plot_grouped,
         subset=df[df.gain_pct < 0].nsmallest(TOP_N, "gain_pct"),
         kwargs=dict(col_b=C_HURT)),

    dict(title="⑧ Largest Plants\n(Max rated_power_kW)",
         fn=plot_power_context,
         subset=df.nlargest(TOP_N, "rated_power_kW")),

    dict(title="⑨ Smallest Plants\n(Min rated_power_kW)",
         fn=plot_power_context,
         subset=df.nsmallest(TOP_N, "rated_power_kW")),
]

# ── Build 3×3 figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16), facecolor="white")
fig.suptitle("MAML Cherry-Pick Analysis — Per-Plant Test Results",
             fontsize=15, fontweight="bold", y=0.995)

gs   = GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.38)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

for ax, panel in zip(axes, panels):
    kwargs = panel.get("kwargs", {})
    if panel["fn"] is plot_grouped:
        plot_grouped(ax, panel["subset"], panel["title"], **kwargs)
    else:
        panel["fn"](ax, panel["subset"], panel["title"])

# ── Shared legend (bottom) ────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_ZERO,  label="Zero-Shot MAE"),
    mpatches.Patch(color=C_ADAPT, label="Adapted MAE"),
    mpatches.Patch(color=C_GAIN,  label="Positive gain annotation"),
    mpatches.Patch(color=C_HURT,  label="Negative gain annotation"),
    mpatches.Patch(color=C_PWR,   alpha=0.6, label="Rated Power (panels ⑧⑨)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

fig.text(0.5, -0.025,
         f"Source: {CSV_PATH}  |  TOP_N per criterion = {TOP_N}",
         ha="center", fontsize=8, color="gray")

# ── Save & show ───────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  💾  Saved → {OUT_PATH}")
plt.show()
