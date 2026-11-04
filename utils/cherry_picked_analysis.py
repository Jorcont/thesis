
"""
Cherry-pick analysis of per-plant MAML test results.
Usage: python cherry_pick_analysis.py path/to/per_plant_test_results.csv [TOP_N]
"""

import sys
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "per_plant_test_results.csv"
TOP_N    = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# ── Load & enrich ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["zero_shot_MAE", "adapted_MAE", "gain_abs", "gain_pct"])

# Normalized MAE (scale-free: MAE per kW)
df["nMAE_zero_shot"] = df["zero_shot_MAE"] / df["rated_power_kW"].replace(0, np.nan)
df["nMAE_adapted"]   = df["adapted_MAE"]   / df["rated_power_kW"].replace(0, np.nan)

COLS = [
    "plant_id", "rated_power_kW",
    "zero_shot_MAE", "adapted_MAE",
    "gain_abs", "gain_pct",
    "nMAE_zero_shot", "nMAE_adapted",
]

SEP  = "─" * 80
SEP2 = "═" * 80

def section(label: str, emoji: str, subset: pd.DataFrame):
    if subset.empty:
        return
    print(f"\n{SEP}")
    print(f"  {emoji}  {label}  (n={len(subset)})")
    print(SEP)
    print(subset[COLS].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print(f"  📊  MAML Cherry-Pick Analysis")
print(f"  Plants: {len(df)}   |   TOP_N: {TOP_N}   |   File: {CSV_PATH}")
print(SEP2)

print(f"\n  {'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"  {'─'*62}")
for col in ["zero_shot_MAE", "adapted_MAE", "gain_abs", "gain_pct", "nMAE_adapted"]:
    print(
        f"  {col:<22} "
        f"{df[col].mean():>10.4f} "
        f"{df[col].std():>10.4f} "
        f"{df[col].min():>10.4f} "
        f"{df[col].max():>10.4f}"
    )

adapt_helped = (df["gain_pct"] > 0).sum()
adapt_hurt   = (df["gain_pct"] < 0).sum()
print(f"\n  Adaptation helped: {adapt_helped}/{len(df)} plants "
      f"({100*adapt_helped/len(df):.1f}%)")
print(f"  Adaptation hurt  : {adapt_hurt}/{len(df)} plants "
      f"({100*adapt_hurt/len(df):.1f}%)")

# ── Cherry-pick criteria ──────────────────────────────────────────────────────
criteria = [
    # Label                                       Emoji  DataFrame slice
    ("Max zero_shot_MAE  – hardest start",        "🔴",  df.nlargest(TOP_N,  "zero_shot_MAE")),
    ("Max gain_pct       – best relative gain",   "🟢",  df[df.gain_pct > 0].nlargest(TOP_N, "gain_pct")),
    ("Max gain_abs       – best absolute gain",   "⬆️ ", df[df.gain_abs > 0].nlargest(TOP_N, "gain_abs")),
    ("Min zero_shot_MAE  – best prior / easiest", "🏆",  df.nsmallest(TOP_N, "zero_shot_MAE")),
    ("Min adapted_MAE    – best final result",    "✅",  df.nsmallest(TOP_N, "adapted_MAE")),
    ("Min nMAE_adapted   – best scale-free perf", "📐",  df.nsmallest(TOP_N, "nMAE_adapted")),
    ("Negative gain      – adaptation HURT",      "⚠️ ", df[df.gain_pct < 0].nsmallest(TOP_N, "gain_pct")),
    ("Largest plants     – high rated power",     "🏭",  df.nlargest(TOP_N,  "rated_power_kW")),
    ("Smallest plants    – low rated power",      "🔋",  df.nsmallest(TOP_N, "rated_power_kW")),
]

for label, emoji, subset in criteria:
    section(label, emoji, subset)

# ── Deduped master cherry-pick table ─────────────────────────────────────────
selected_ids = set()
for _, _, subset in criteria:
    selected_ids.update(subset["plant_id"].tolist())

cherry = (
    df[df["plant_id"].isin(selected_ids)]
    .sort_values("gain_pct", ascending=False)
    .reset_index(drop=True)
)

print(f"\n{SEP2}")
print(f"  🍒  Unique cherry-picked plants  ({len(cherry)} total)")
print(SEP2)
print(cherry[COLS].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# ── Optional CSV export ───────────────────────────────────────────────────────
out_path = CSV_PATH.replace(".csv", "_cherry_picked.csv")
cherry.to_csv(out_path, index=False)
print(f"\n  💾  Saved → {out_path}\n")
