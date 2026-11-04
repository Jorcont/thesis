"""
Utility to create a minimal TimeSeriesDataSet for TemporalFusionTransformer
model instantiation via from_dataset().

This dataset is used ONLY for architecture configuration (column names, sizes).
Actual training data comes from PVTimeSeriesDatasetV2 (Arrow pre-load).
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict


def create_minimal_pf_dataset(
    time_varying_known_reals: List[str],
    time_varying_unknown_reals: List[str],
    target: str,
    max_encoder_length: int = 96,
    max_prediction_length: int = 24,
    static_reals: Optional[List[str]] = None,
):
    """
    Build a minimal TimeSeriesDataSet for model construction via from_dataset().

    This creates a small synthetic DataFrame with the correct column structure.
    The TFT reads column metadata from it — no real data needed.

    Args:
        time_varying_known_reals:  Feature names known in advance (NWP + solar + time)
        time_varying_unknown_reals: Features only observed in past (typically [target])
        target: Target column name
        max_encoder_length: Look-back window size
        max_prediction_length: Look-ahead window size

    Returns:
        TimeSeriesDataSet instance (for from_dataset() only)
    """
    from pytorch_forecasting import TimeSeriesDataSet

    static_reals = static_reals or []

    # ── Create synthetic DataFrame with correct schema ─────────────
    # Need at least (encoder + prediction) * 2 rows for valid dataset
    n_rows = (max_encoder_length + max_prediction_length) * 3

    data = {
        "time_idx": np.arange(n_rows),
        "group_id": ["plant_0"] * n_rows,  # dummy group
    }

    # Known reals: random values (schema only matters, not values)
    for col in time_varying_known_reals:
        data[col] = np.random.randn(n_rows).astype(np.float32)

    # Unknown reals (includes target)
    for col in time_varying_unknown_reals:
        if col not in data:
            data[col] = np.random.rand(n_rows).astype(np.float32)

    # Ensure target exists
    if target not in data:
        data[target] = np.random.rand(n_rows).astype(np.float32)

    # ── FIX: populate static real columns so the DataFrame has them ──
    for col in static_reals:
        if col not in data:
            data[col] = np.random.randn(n_rows).astype(np.float32)
    # ─────────────────────────────────────────────────────────────────


    df = pd.DataFrame(data)

    # ── Create TimeSeriesDataSet ──────────────────────────────────
    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=["group_id"],
        static_reals=static_reals,
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        target_normalizer=None,  # our data is already normalized
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )

    return dataset


def verify_pf_dict_format(dataset) -> Dict[str, tuple]:
    """
    Extract ONE sample from the TimeSeriesDataSet and print its structure.
    Use this to verify that _to_pf_dict() produces matching shapes.

    Args:
        dataset: TimeSeriesDataSet instance from create_minimal_pf_dataset()

    Returns:
        Dict mapping key → shape tuple
    """
    from torch.utils.data import DataLoader

    dl = dataset.to_dataloader(batch_size=2, shuffle=False, num_workers=0)
    x_dict, (y, weight) = next(iter(dl))

    print("\n" + "=" * 60)
    print("[verify_pf_dict_format] pytorch_forecasting TFT input dict:")
    print("=" * 60)

    shapes = {}
    for key, val in x_dict.items():
        if isinstance(val, torch.Tensor):
            shapes[key] = tuple(val.shape)
            print(f"  {key:25s} : {val.dtype}  shape={val.shape}")
        elif isinstance(val, (list, tuple)):
            print(f"  {key:25s} : list/tuple len={len(val)}")
            shapes[key] = (len(val),)
        else:
            print(f"  {key:25s} : {type(val).__name__} = {val}")
            shapes[key] = None

    print(f"\n  y                       : {y.dtype}  shape={y.shape}")
    print("=" * 60 + "\n")

    return shapes