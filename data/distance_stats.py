import pandas as pd
import sys

sys.path.insert(0, "/mnt/data/kes/data/jdomingo/meta-netze")
from data.parquet_loader import load_plant_nwp_mapping

# ── Load mapping ──────────────────────────────────────────────
mapping = load_plant_nwp_mapping()

# ── Distance stats ────────────────────────────────────────────
stats = mapping["dist_km"].agg(["min", "max", "mean", "median", "std"])

print("=" * 50)
print("  Distance Stats: PV Plants → NWP Tiles (km)")
print("=" * 50)
print(f"  Min    : {stats['min']:.3f} km")
print(f"  Max    : {stats['max']:.3f} km")
print(f"  Mean   : {stats['mean']:.3f} km")
print(f"  Median : {stats['median']:.3f} km")
print(f"  Std    : {stats['std']:.3f} km")
print("=" * 50)

# ── Refined distribution (focused on < 5 km) ─────────────────
bins   = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, float("inf")]
labels = [
    "< 0.5 km",
    "0.5–1.0 km",
    "1.0–1.5 km",
    "1.5–2.0 km",
    "2.0–2.5 km",
    "2.5–3.0 km",
    "3.0–3.5 km",
    "3.5–4.0 km",
    "4.0–4.5 km",
    "> 4.5 km",
]
mapping["dist_bin"] = pd.cut(mapping["dist_km"], bins=bins, labels=labels)

print("\n  Distribution:")
dist_table = mapping["dist_bin"].value_counts().sort_index()
for label, count in dist_table.items():
    pct     = count / len(mapping) * 100
    bar     = "█" * int(pct / 2)   # visual bar
    print(f"  {label:<14}: {count:>5} plants  ({pct:>5.1f}%)  {bar}")

print("=" * 50)
print(f"  Total plants : {len(mapping)}")
print(f"  Unique tiles : {mapping['nwp_tile'].nunique()}")
print("=" * 50)
