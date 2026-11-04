# #!/usr/bin/env python3
# """
# Cherry-pick visual analysis — 3x2 plot grid (6 panels).

# Usable as:
#   CLI:
#     python cherry_pick_plots.py path/to/per_plant_test_results.csv [TOP_N]
#   Module:
#     from cherry_pick_plots import make_cherry_pick_grid
#     make_cherry_pick_grid(df, top_n=10, out_path="out.png")

# Expected DataFrame columns (plain or *_mean naming accepted):
#   plant_id, zero_shot_MAE, adapted_MAE, gain_abs, gain_pct
# """

# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec
# from pathlib import Path

# # ── Palette ───────────────────────────────────────────────────────────────────
# C_ZERO  = "#E07B54"
# C_ADAPT = "#4C9BE8"
# C_GAIN  = "#5DBB63"
# C_HURT  = "#E05454"
# C_BG    = "#F8F9FA"
# C_GRID  = "#DEDEDE"


# # ── Column normalisation ──────────────────────────────────────────────────────
# def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Accept 'zero_shot_MAE_mean'/'adapted_MAE_mean' (seed-averaged)
#     in place of 'zero_shot_MAE'/'adapted_MAE' (single-seed)."""
#     mapping = {}
#     if "zero_shot_MAE" not in df.columns and "zero_shot_MAE_mean" in df.columns:
#         mapping["zero_shot_MAE_mean"] = "zero_shot_MAE"
#     if "adapted_MAE" not in df.columns and "adapted_MAE_mean" in df.columns:
#         mapping["adapted_MAE_mean"] = "adapted_MAE"
#     return df.rename(columns=mapping)


# def short_id(pid, n=3):
#     s = str(pid)
#     return f"…{s[-n:]}" if len(s) > n else s


# # ── Shared drawing helpers ────────────────────────────────────────────────────
# def _style(ax, title):
#     ax.set_title(title, fontsize=9, fontweight="bold", pad=7)
#     ax.set_facecolor(C_BG)
#     ax.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
#     ax.spines[["top", "right"]].set_visible(False)
#     ax.tick_params(labelsize=8)


# def _annotate_gain(ax, x_pos, y_pos, gain_pct):
#     color = C_GAIN if gain_pct >= 0 else C_HURT
#     sign  = "+" if gain_pct >= 0 else ""
#     ax.annotate(f"{sign}{gain_pct:.1f}%",
#                 xy=(x_pos, y_pos), fontsize=7.5,
#                 ha="center", va="bottom",
#                 color=color, fontweight="bold")


# def _empty(ax, title):
#     ax.set_title(title, fontsize=9, fontweight="bold", pad=7)
#     ax.set_facecolor(C_BG)
#     ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.text(0.5, 0.5, "No data for\nthis criterion",
#             ha="center", va="center", transform=ax.transAxes,
#             fontsize=10, color="gray", style="italic")


# def plot_grouped(ax, subset, title,
#                  col_a=C_ZERO, col_b=C_ADAPT,
#                  ycol_a="zero_shot_MAE", ycol_b="adapted_MAE",
#                  label_a="Zero-Shot", label_b="Adapted",
#                  ylabel="MAE"):
#     sub = subset.reset_index(drop=True)
#     if sub.empty:
#         _empty(ax, title); return

#     n, w = len(sub), 0.35
#     x = np.arange(n)

#     ax.bar(x - w/2, sub[ycol_a], w, color=col_a, label=label_a,
#            edgecolor="white", linewidth=0.8, zorder=3)
#     ax.bar(x + w/2, sub[ycol_b], w, color=col_b, label=label_b,
#            edgecolor="white", linewidth=0.8, zorder=3)

#     ax.set_xticks(x)
#     ax.set_xticklabels([short_id(p) for p in sub["plant_id"]],
#                        rotation=30, ha="right", fontsize=11)
#     ax.set_ylabel(ylabel, fontsize=8)
#     ax.legend(fontsize=7, framealpha=0.7)
#     _style(ax, title)

#     ymax = max(sub[ycol_a].max(), sub[ycol_b].max())
#     for i, row in sub.iterrows():
#         y_top = max(row[ycol_a], row[ycol_b]) + ymax * 0.03
#         _annotate_gain(ax, x[i], y_top, row["gain_pct"])


# # ── Main entry point (callable) ────────────────────────────────────────────────
# def make_cherry_pick_grid(df: pd.DataFrame,
#                           top_n: int = 10,
#                           out_path=None,
#                           title: str = "Cherry-Pick Analysis — Per-Plant Test Results",
#                           source_label: str = None,
#                           show: bool = False,
#                           dpi: int = 300):
#     """
#     Build the 2×3 cherry-pick figure (6 panels) from a per-plant DataFrame.

#     Parameters
#     ----------
#     df : DataFrame with plant_id, zero_shot_MAE, adapted_MAE, gain_abs,
#          gain_pct (plain or *_mean naming accepted).
#     top_n : plants per panel.
#     out_path : save target; parent dirs created as needed. None = do not save.
#     source_label : text shown in the footer; defaults to `out_path`.
#     show : if True, calls plt.show() (interactive use only).
#     """
#     df = _normalize_columns(df).copy()
#     df = df.dropna(subset=["zero_shot_MAE", "adapted_MAE", "gain_abs", "gain_pct"])

#     panels = [
#         dict(title="① Hardest Start\n(Max Zero-Shot MAE)",
#              subset=df.nlargest(top_n, "zero_shot_MAE")),
#         dict(title="② Best Relative Gain\n(Max gain_pct)",
#              subset=df[df.gain_pct > 0].nlargest(top_n, "gain_pct")),
#         dict(title="③ Best Absolute Gain\n(Max gain_abs)",
#              subset=df[df.gain_abs > 0].nlargest(top_n, "gain_abs")),
#         dict(title="④ Easiest Start\n(Min Zero-Shot MAE)",
#              subset=df.nsmallest(top_n, "zero_shot_MAE")),
#         dict(title="⑤ Best Final Result\n(Min Adapted MAE)",
#              subset=df.nsmallest(top_n, "adapted_MAE")),
#         dict(title="⑥ Adaptation HURT\n(Most Negative gain_pct)",
#              subset=df[df.gain_pct < 0].nsmallest(top_n, "gain_pct"),
#              kwargs=dict(col_b=C_HURT)),
#     ]

#     n_rows, n_cols = 3, 2
#     fig = plt.figure(figsize=(22, 11), facecolor="white")
#     fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

#     gs   = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.28)
#     axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(len(panels))]

#     for ax, panel in zip(axes, panels):
#         plot_grouped(ax, panel["subset"], panel["title"], **panel.get("kwargs", {}))

#     legend_handles = [
#         mpatches.Patch(color=C_ZERO,  label="Zero-Shot MAE"),
#         mpatches.Patch(color=C_ADAPT, label="Adapted MAE"),
#         mpatches.Patch(color=C_GAIN,  label="Positive gain annotation"),
#         mpatches.Patch(color=C_HURT,  label="Negative gain annotation"),
#     ]
#     fig.legend(handles=legend_handles, loc="lower center", ncol=4,
#                fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

#     footer = source_label if source_label is not None else (str(out_path) if out_path else "")
#     if footer:
#         fig.text(0.5, -0.025,
#                  f"Source: {footer}  |  TOP_N per criterion = {top_n}",
#                  ha="center", fontsize=8, color="gray")

#     if out_path is not None:
#         out_path = Path(out_path)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
#         print(f"  💾  Saved → {out_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

#     return fig


# def main():
#     csv_path = sys.argv[1] if len(sys.argv) > 1 else "per_plant_test_results.csv"
#     top_n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
#     out_path = Path(csv_path).parent / "cherry_pick_plots.png"

#     df = pd.read_csv(csv_path)
#     make_cherry_pick_grid(df, top_n=top_n, out_path=out_path,
#                           source_label=str(csv_path), show=True)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Cherry-pick visual analysis — 3x2 plot grid (6 panels).

Usable as:
  CLI:
    python cherry_pick_plots.py path/to/per_plant_test_results.csv [TOP_N]
  Module:
    from cherry_pick_plots import make_cherry_pick_grid
    make_cherry_pick_grid(df, top_n=10, out_path="out.png")

Expected DataFrame columns (plain or *_mean naming accepted):
  plant_id, zero_shot_MAE, adapted_MAE, gain_abs, gain_pct
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Palette ───────────────────────────────────────────────────────────────────
C_ZERO  = "#E07B54"
C_ADAPT = "#4C9BE8"
C_GAIN  = "#5DBB63"
C_HURT  = "#E05454"
C_BG    = "#F8F9FA"
C_GRID  = "#DEDEDE"


# ── Column normalisation ──────────────────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'zero_shot_MAE_mean'/'adapted_MAE_mean' (seed-averaged)
    in place of 'zero_shot_MAE'/'adapted_MAE' (single-seed)."""
    mapping = {}
    if "zero_shot_MAE" not in df.columns and "zero_shot_MAE_mean" in df.columns:
        mapping["zero_shot_MAE_mean"] = "zero_shot_MAE"
    if "adapted_MAE" not in df.columns and "adapted_MAE_mean" in df.columns:
        mapping["adapted_MAE_mean"] = "adapted_MAE"
    return df.rename(columns=mapping)


def short_id(pid, n=3):
    s = str(pid)
    return f"…{s[-n:]}" if len(s) > n else s


# ── Shared drawing helpers ────────────────────────────────────────────────────
def _style(ax, title):
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=14)


def _annotate_gain(ax, x_pos, y_pos, gain_pct):
    color = C_GAIN if gain_pct >= 0 else C_HURT
    sign  = "+" if gain_pct >= 0 else ""
    ax.annotate(f"{sign}{gain_pct:.1f}%",
                xy=(x_pos, y_pos), fontsize=12.5,
                ha="center", va="bottom",
                color=color, fontweight="bold")


def _empty(ax, title):
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_facecolor(C_BG)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, "No data for\nthis criterion",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=16, color="gray", style="italic")


def plot_grouped(ax, subset, title,
                 col_a=C_ZERO, col_b=C_ADAPT,
                 ycol_a="zero_shot_MAE", ycol_b="adapted_MAE",
                 label_a="Zero-Shot", label_b="Adapted",
                 ylabel="MAE"):
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
                       rotation=30, ha="right", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.legend(fontsize=12, framealpha=0.7)
    _style(ax, title)

    ymax = max(sub[ycol_a].max(), sub[ycol_b].max())
    for i, row in sub.iterrows():
        y_top = max(row[ycol_a], row[ycol_b]) + ymax * 0.03
        _annotate_gain(ax, x[i], y_top, row["gain_pct"])


# ── Main entry point (callable) ────────────────────────────────────────────────
def make_cherry_pick_grid(df: pd.DataFrame,
                          top_n: int = 10,
                          out_path=None,
                          title: str = "Cherry-Pick Analysis — Per-Plant Test Results",
                          source_label: str = None,
                          show: bool = False,
                          dpi: int = 300):
    """
    Build the 2×3 cherry-pick figure (6 panels) from a per-plant DataFrame.

    Parameters
    ----------
    df : DataFrame with plant_id, zero_shot_MAE, adapted_MAE, gain_abs,
         gain_pct (plain or *_mean naming accepted).
    top_n : plants per panel.
    out_path : save target; parent dirs created as needed. None = do not save.
    source_label : text shown in the footer; defaults to `out_path`.
    show : if True, calls plt.show() (interactive use only).
    """
    df = _normalize_columns(df).copy()
    df = df.dropna(subset=["zero_shot_MAE", "adapted_MAE", "gain_abs", "gain_pct"])

    panels = [
        dict(title="(a) Lowest Zero-Shot Error\n(Minimum Zero-Shot MAE)",
             subset=df.nsmallest(top_n, "zero_shot_MAE")),
        dict(title="(b) Highest Zero-Shot Error\n(Maximum Zero-Shot MAE)",
             subset=df.nlargest(top_n, "zero_shot_MAE")),
        dict(title="(c) Largest Absolute Gain\n(Maximum gain_abs)",
             subset=df[df.gain_abs > 0].nlargest(top_n, "gain_abs")),
        dict(title="(d) Largest Relative Gain\n(Maximum gain_pct)",
             subset=df[df.gain_pct > 0].nlargest(top_n, "gain_pct")),
        dict(title="(e) Best Adapted Performance\n(Minimum Adapted MAE)",
             subset=df.nsmallest(top_n, "adapted_MAE")),
        dict(title="(f) Negative Adaptation Effect\n(Largest Decrease in Performance)",
             subset=df[df.gain_pct < 0].nsmallest(top_n, "gain_pct"),
             kwargs=dict(col_b=C_HURT)),
    ]

    n_rows, n_cols = 3, 2
    # Increased figure size to give more room for larger text/annotations
    fig = plt.figure(figsize=(28, 16), facecolor="white")
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)

    gs   = GridSpec(n_rows, n_cols, figure=fig, hspace=0.65, wspace=0.32)
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(len(panels))]

    for ax, panel in zip(axes, panels):
        plot_grouped(ax, panel["subset"], panel["title"], **panel.get("kwargs", {}))

    legend_handles = [
        mpatches.Patch(color=C_ZERO,  label="Zero-Shot MAE"),
        mpatches.Patch(color=C_ADAPT, label="Adapted MAE"),
        mpatches.Patch(color=C_GAIN,  label="Positive gain annotation"),
        mpatches.Patch(color=C_HURT,  label="Negative gain annotation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=28, framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    footer = source_label if source_label is not None else (str(out_path) if out_path else "")
    if footer:
        fig.text(0.5, -0.035,
                 f"Source: {footer}  |  TOP_N per criterion = {top_n}",
                 ha="center", fontsize=12, color="gray")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  💾  Saved → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "per_plant_test_results.csv"
    top_n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out_path = Path(csv_path).parent / "cherry_pick_plots.png"

    df = pd.read_csv(csv_path)
    make_cherry_pick_grid(df, top_n=top_n, out_path=out_path,
                          source_label=str(csv_path), show=True)


if __name__ == "__main__":
    main()