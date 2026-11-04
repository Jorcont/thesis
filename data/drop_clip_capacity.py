#!/usr/bin/env python3
"""
drop_clip_capacity.py

Identifies and (optionally) removes PV plants whose renormalized power curve is
corrupted by capacity errors, i.e. clipping to 1.0 flattens a large part of the
real signal.

Steps:
  1. Compute clipped_pct for every candidate (new_max > 1.2).
  2. Apply the drop policy:
        - new_max > 2.0                              -> drop (irrecoverable)
        - 1.2 < new_max <= 2.0 and clipped_pct > 1%  -> drop (heavy clipping)
        - otherwise                                  -> keep
  3. Merge dropped plants into _excluded_plants.csv and delete their parquet
     files.  Guarded by DRY_RUN (default True) -> nothing is written/deleted.

Usage:
    python drop_clip_capacity.py            # dry run (report only)
    python drop_clip_capacity.py --apply    # actually delete + merge
"""

import os
import sys
import glob
import argparse
import pandas as pd

# ────────────────────────── Config ──────────────────────────
CLEAN_DIR   = "/mnt/data/kes/data/jdomingo/plants_clean"
REPORT_CSV  = os.path.join(CLEAN_DIR, "_renorm_report.csv")
EXCLUDE_CSV = os.path.join(CLEAN_DIR, "_excluded_plants.csv")

CAND_MIN_MAX   = 1.2   # only inspect plants above this new_max
HARD_MAX       = 2.0   # new_max above this -> drop unconditionally
MID_CLIP_PCT   = 1.0   # in the 1.2-2.0 band, drop if clipped_pct exceeds this

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ───────────────────── Step 1: clip percentage ─────────────────────
def compute_clip_stats() -> pd.DataFrame:
    rep = pd.read_csv(REPORT_CSV)
    cand = rep[rep["new_max"] > CAND_MIN_MAX].copy()
    print(f"[Step 1] Candidates (new_max > {CAND_MIN_MAX}): {len(cand)}")

    rows, missing = [], 0
    for _, r in cand.iterrows():
        pid   = int(r["plant_id"])
        fpath = os.path.join(CLEAN_DIR, f"{pid}.parquet")
        if not os.path.exists(fpath):
            missing += 1
            continue
        ap = pd.read_parquet(fpath, columns=["active_power_kW"])["active_power_kW"]
        n_valid = int(ap.notna().sum())
        n_clip  = int(r["clipped_rows"])
        rows.append({
            "plant_id":     pid,
            "new_max":      float(r["new_max"]),
            "valid_rows":   n_valid,
            "clipped_rows": n_clip,
            "clipped_pct":  (100.0 * n_clip / n_valid) if n_valid else float("nan"),
        })

    a = pd.DataFrame(rows).sort_values("clipped_pct", ascending=False)
    print(f"[Step 1] Parquet files missing for candidates: {missing}")

    print("\n[Step 1] clipped_pct distribution:")
    print(a["clipped_pct"].describe().to_string())

    bands = pd.cut(a["clipped_pct"],
                   bins=[-0.001, 0.1, 0.5, 1.0, 5.0, 100.0],
                   labels=["<=0.1%", "0.1-0.5%", "0.5-1%", "1-5%", ">5%"])
    nm = pd.cut(a["new_max"], bins=[1.2, 2.0, 5.0], labels=["1.2-2.0", "2.0-5.0"])
    print("\n[Step 1] Cross-tab (new_max band x clip severity):")
    print(pd.crosstab(nm, bands).to_string())
    return a


# ───────────────────── Step 2: apply policy ─────────────────────
def apply_policy(a: pd.DataFrame) -> pd.DataFrame:
    drop_high_max = a["new_max"] > HARD_MAX
    drop_mid_clip = (a["new_max"] <= HARD_MAX) & (a["clipped_pct"] > MID_CLIP_PCT)

    a["decision"] = "keep"
    a.loc[drop_high_max, "decision"] = f"drop:new_max>{HARD_MAX}"
    a.loc[drop_mid_clip, "decision"] = f"drop:clip>{MID_CLIP_PCT}%"

    print("\n[Step 2] Decision breakdown:")
    print(a["decision"].value_counts().to_string())
    n_drop = int((a["decision"] != "keep").sum())
    n_keep = int((a["decision"] == "keep").sum())
    print(f"\n[Step 2] Total to DROP: {n_drop}")
    print(f"[Step 2] Total to KEEP: {n_keep}")

    print("\n[Step 2] Top kept borderline plants (should look clean):")
    print(a[a["decision"] == "keep"]
            .sort_values("clipped_pct", ascending=False)
            .head(10).to_string(index=False))
    return a


# ───────────────────── Step 3: merge + delete ─────────────────────
def apply_changes(a: pd.DataFrame, dry_run: bool):
    to_drop = a[a["decision"] != "keep"].copy()
    if to_drop.empty:
        print("\n[Step 3] Nothing to drop. Exiting.")
        return

    # Build rows matching the existing _excluded_plants.csv schema:
    # plant_id, reason, cap_sum   (cap_sum taken from _renorm_report.csv)
    rep = pd.read_csv(REPORT_CSV)[["plant_id", "cap_sum"]]
    new_exc = to_drop.merge(rep, on="plant_id", how="left")
    new_exc = new_exc.rename(columns={"decision": "reason"})[
        ["plant_id", "reason", "cap_sum"]
    ]

    mode = "DRY RUN" if dry_run else "APPLYING"
    print(f"\n[Step 3] ({mode}) Plants to drop: {len(new_exc)}")
    print(new_exc.head(10).to_string(index=False))

    # ---- 3a. Merge into _excluded_plants.csv ----
    if os.path.exists(EXCLUDE_CSV):
        exc = pd.read_csv(EXCLUDE_CSV)
    else:
        exc = pd.DataFrame(columns=["plant_id", "reason", "cap_sum"])

    before = len(exc)
    already = set(exc["plant_id"].astype(int))
    new_exc = new_exc[~new_exc["plant_id"].isin(already)]
    merged = pd.concat([exc, new_exc], ignore_index=True)
    print(f"\n[Step 3] _excluded_plants.csv: {before} -> {len(merged)} rows "
          f"(+{len(new_exc)} new)")

    # ---- 3b. Delete parquet files ----
    missing = 0
    for pid in new_exc["plant_id"].astype(int):
        fpath = os.path.join(CLEAN_DIR, f"{pid}.parquet")
        if not os.path.exists(fpath):
            missing += 1
            continue
        if dry_run:
            print(f"    [dry-run] would delete {fpath}")
        else:
            os.remove(fpath)
    if missing:
        print(f"[Step 3] (warn) {missing} parquet files already absent.")

    # ---- 3c. Persist CSV ----
    if dry_run:
        print("\n[Step 3] DRY RUN complete — no files written or deleted.")
        print("         Re-run with --apply to commit these changes.")
    else:
        # safety backup of the CSV before overwriting
        if os.path.exists(EXCLUDE_CSV):
            bak = EXCLUDE_CSV + ".bak"
            pd.read_csv(EXCLUDE_CSV).to_csv(bak, index=False)
            print(f"[Step 3] Backup written: {bak}")
        merged.to_csv(EXCLUDE_CSV, index=False)
        remaining = len(glob.glob(os.path.join(CLEAN_DIR, "*.parquet")))
        print(f"[Step 3] DONE. Deleted {len(new_exc)} parquets. "
              f"Remaining parquet files: {remaining}")


# ────────────────────────── Main ──────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete parquets and update the CSV. "
                             "Default is a dry run.")
    args = parser.parse_args()
    dry_run = not args.apply

    print("=" * 70)
    print(f"drop_clip_capacity.py  |  mode: {'DRY RUN' if dry_run else 'APPLY'}")
    print("=" * 70)

    a = compute_clip_stats()
    a = apply_policy(a)
    apply_changes(a, dry_run=dry_run)


if __name__ == "__main__":
    main()