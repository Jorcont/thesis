"""
Map each trafo to its nearest NWP grid tile via Haversine distance.
Reads:  master_data_merged.csv, grid_coords.parquet
Writes: trafo_nwp_mapping.parquet
"""

import os
import numpy as np
import pandas as pd

MASTER_CSV  = ("/mnt/data/kes/data/jdomingo/trafo_raw_data/"
               "masterdata/master_data_merged.csv")
NWP_PARQUET = "/mnt/data/kes/data/jdomingo/trafo_nwp_parquet"
OUT_PATH    = f"{NWP_PARQUET}/trafo_nwp_mapping.parquet"


def haversine_nearest(lat: float, lon: float,
                      grid_df: pd.DataFrame) -> tuple[str, float]:
    """Return (tile_id, distance_km) for the closest NWP tile."""
    lat_r = np.radians(lat)
    dlat  = np.radians(grid_df["lats"] - lat)
    dlon  = np.radians(grid_df["lons"] - lon)
    a     = (np.sin(dlat / 2) ** 2
             + np.cos(lat_r) * np.cos(np.radians(grid_df["lats"]))
             * np.sin(dlon / 2) ** 2)
    dist  = 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    idx   = dist.idxmin()
    return idx, round(float(dist[idx]), 3)


def run():
    master  = pd.read_csv(MASTER_CSV)
    grid_df = pd.read_parquet(f"{NWP_PARQUET}/grid_coords.parquet")

    # ── Filter grid to only tiles with available parquet data ──────────────
    available_tiles = {
        f.replace(".parquet", "")
        for f in os.listdir(NWP_PARQUET)
        if f.endswith(".parquet")
        and f not in ("grid_coords.parquet", "trafo_nwp_mapping.parquet")
    }
    grid_df = grid_df[grid_df.index.isin(available_tiles)]
    # ───────────────────────────────────────────────────────────────────────

    print(f"Trafos          : {len(master)}")
    print(f"NWP tiles total : {len(pd.read_parquet(f'{NWP_PARQUET}/grid_coords.parquet'))}")
    print(f"NWP tiles avail : {len(grid_df)}  ← search restricted to these")

    results = master.apply(
        lambda r: haversine_nearest(r["latitude"], r["longitude"], grid_df),
        axis=1,
        result_type="expand",
    )
    master["nwp_tile"] = results[0]
    master["dist_km"]  = results[1]

    print(f"\nDistance stats (km):")
    print(f"  mean  : {master['dist_km'].mean():.2f}")
    print(f"  max   : {master['dist_km'].max():.2f}")
    print(f"  > 10  : {(master['dist_km'] > 10).sum()} trafos")

    cols = ["asset_id", "latitude", "longitude", "nwp_tile", "dist_km",
            "voltage_level_upper", "voltage_level_lower", "dso_name"]
    master[cols].to_parquet(OUT_PATH, index=False)
    print(f"\n✅ Saved → trafo_nwp_mapping.parquet")


if __name__ == "__main__":
    run()
