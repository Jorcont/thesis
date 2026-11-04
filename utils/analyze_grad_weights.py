#!/usr/bin/env python3
"""
analyze_grad_weights.py
────────────────────────────────────────────────────────────────────────────
Monitor weight-norms and grad-norms from TFT training .out log files.

Usage
─────
  python utils/analyze_grad_weights.py output/logs/tft_v2_218330.out
  python utils/analyze_grad_weights.py output/logs/tft_v2_218330.out --no-plot
  python utils/analyze_grad_weights.py output/logs/tft_v2_*.out      # multi-file
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker


# ─── Regex ────────────────────────────────────────────────────────────────────
RE_WEIGHT = re.compile(r'^\s*weight\s+([\w.]+)\s*:\s*([\d.eE+-]+)')
RE_GRAD   = re.compile(r'^([\w.]+):\s*grad_norm=([\d.eE+-]+)')
RE_EPOCH  = re.compile(r'(?:epoch|Epoch|EPOCH)\s*[=:_]?\s*(\d+)')

COLORS = plt.cm.tab10.colors


# ─── Component grouping ───────────────────────────────────────────────────────
def classify(param: str) -> str:
    if 'vsn_enc'       in param: return 'vsn_enc'
    if 'vsn_dec'       in param: return 'vsn_dec'
    if '_proj'         in param: return 'proj'
    if 'encoder_grn'   in param: return 'encoder_grn'
    if 'encoder'       in param: return 'encoder_lstm'
    if 'post_attn_grn' in param: return 'post_attn'
    if 'attn_norm'     in param: return 'post_attn'
    if 'attn'          in param: return 'attention'
    if param.startswith('head'): return 'head'
    return 'other'


# ─── Parser ───────────────────────────────────────────────────────────────────
def parse_file(path: Path) -> pd.DataFrame:
    """
    Parse .out file and return DataFrame with columns:
      snapshot, epoch, param, component, weight_norm, grad_norm, grad_weight_ratio
    A new snapshot is detected when weight lines reappear after grad lines.
    """
    weights:    dict = {}
    grads:      dict = {}
    snap_epoch: dict = {}

    snapshot  = 0
    epoch     = None
    last_type = None  # 'w' or 'g'

    with open(path, errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip()

            m = RE_EPOCH.search(line)
            if m:
                epoch = int(m.group(1))

            m = RE_WEIGHT.match(line)
            if m:
                if last_type == 'g':       # transition grad→weight = new snapshot
                    snapshot += 1
                snap_epoch.setdefault(snapshot, epoch)
                weights[(snapshot, m.group(1))] = float(m.group(2))
                last_type = 'w'
                continue

            m = RE_GRAD.match(line)
            if m:
                snap_epoch.setdefault(snapshot, epoch)
                grads[(snapshot, m.group(1))] = float(m.group(2))
                last_type = 'g'

    all_params    = sorted({k[1] for k in weights} | {k[1] for k in grads})
    all_snapshots = sorted({k[0] for k in weights} | {k[0] for k in grads})

    rows = []
    for snap in all_snapshots:
        for param in all_params:
            w = weights.get((snap, param), float('nan'))
            g = grads.get((snap, param),   float('nan'))
            if not (np.isnan(w) and np.isnan(g)):
                rows.append({
                    'snapshot':    snap,
                    'epoch':       snap_epoch.get(snap),
                    'param':       param,
                    'component':   classify(param),
                    'weight_norm': w,
                    'grad_norm':   g,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['grad_weight_ratio'] = (
            df['grad_norm'] / df['weight_norm'].replace(0, np.nan)
        )
    return df


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_by_component(df: pd.DataFrame, metric: str, title: str, out_path: Path):
    """Mean metric per component over snapshots."""
    pivot = (
        df.groupby(['snapshot', 'component'])[metric]
        .mean()
        .unstack('component')
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[col], marker='o', markersize=3,
                label=col, color=COLORS[i % len(COLORS)], linewidth=1.5)

    ax.set_xlabel('Snapshot')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  → {out_path}')


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, out_path: Path):
    """Heatmap: param × snapshot (log-scaled color)."""
    pivot = df.pivot_table(
        index='param', columns='snapshot', values=metric, aggfunc='mean'
    )
    if pivot.empty:
        return

    vals = pivot.values
    vmin = max(1e-9, np.nanmin(vals[vals > 0])) if np.any(vals > 0) else 1e-9
    vmax = np.nanmax(vals) if not np.all(np.isnan(vals)) else 1.0

    h = max(8,  len(pivot)          * 0.25)
    w = max(10, len(pivot.columns)  * 0.55)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(
        vals, aspect='auto', cmap='viridis',
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.6, label=metric)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  → {out_path}')


def plot_ratio_heatmap(df: pd.DataFrame, title: str, out_path: Path):
    """Grad/weight ratio heatmap — highlights vanishing/exploding gradients."""
    plot_heatmap(df, 'grad_weight_ratio', title, out_path)


# ─── Terminal summary ─────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, version: str):
    last = df[df['snapshot'] == df['snapshot'].max()]

    print(f'\n{"═"*72}')
    print(f'  📊 Last-snapshot summary — {version}')
    print(f'{"═"*72}')

    tbl = (
        last.groupby('component')[['weight_norm', 'grad_norm', 'grad_weight_ratio']]
        .agg(['mean', 'max'])
        .round(6)
    )
    print(tbl.to_string())

    # Dead parameters
    print(f'\n{"─"*72}')
    print('  ⚠️  Dead parameters  (grad_norm < 1e-6)')
    dead = last[last['grad_norm'] < 1e-6]
    if dead.empty:
        print('  ✅ None found')
    else:
        for _, row in dead.iterrows():
            print(f'  ❌  {row.param:<52s}  grad={row.grad_norm:.2e}')

    # Top grad norms
    print(f'\n{"─"*72}')
    print('  🔥 Largest grad_norms (top 20)')
    top = last.nlargest(20, 'grad_norm')[
        ['param', 'component', 'grad_norm', 'weight_norm', 'grad_weight_ratio']
    ]
    print(top.to_string(index=False))
    print(f'{"═"*72}\n')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files',    nargs='+', type=Path)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=Path('utils/output'))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.files:
        if not path.exists():
            print(f'⚠️  Not found: {path}', file=sys.stderr)
            continue

        version = path.stem
        print(f'\n{"═"*72}')
        print(f'  📂  {path}')
        print(f'{"═"*72}')

        df = parse_file(path)
        if df.empty:
            print('  ⚠️  No weight/grad data found — check log format.')
            continue

        print(f'  ℹ️  Snapshots: {df["snapshot"].nunique()} | '
              f'Parameters: {df["param"].nunique()}')

        # CSV
        csv = args.out_dir / f'{version}_grad_weights.csv'
        df.to_csv(csv, index=False)
        print(f'  → CSV: {csv}')

        print_summary(df, version)

        if not args.no_plot:
            pfx = args.out_dir / version
            plot_by_component(df, 'grad_norm',
                f'Grad Norm by Component — {version}',
                Path(f'{pfx}_grad_norm.png'))
            plot_by_component(df, 'weight_norm',
                f'Weight Norm by Component — {version}',
                Path(f'{pfx}_weight_norm.png'))
            plot_by_component(df, 'grad_weight_ratio',
                f'Grad/Weight Ratio by Component — {version}',
                Path(f'{pfx}_grad_weight_ratio.png'))
            plot_heatmap(df, 'grad_norm',
                f'Grad Norm Heatmap — {version}',
                Path(f'{pfx}_grad_norm_heatmap.png'))
            plot_heatmap(df, 'grad_weight_ratio',
                f'Grad/Weight Ratio Heatmap — {version}',
                Path(f'{pfx}_ratio_heatmap.png'))


if __name__ == '__main__':
    main()
