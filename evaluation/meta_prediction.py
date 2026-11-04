import torch
import numpy as np
import pandas as pd
from typing import List, Union


class MetaPrediction:
    """
    Translates TFTRegressor output tensors → result_dict format for TFTTester.
    Supports both single-target and multi-target forecasting
    result_dict format:
    {
        0.1: [ DataFrame_window_0, DataFrame_window_1, ... ],
        0.5: [ ... ],
        0.9: [ ... ],
    }
    Each DataFrame columns:
        target_1/Predicted, target_2/Predicted, ..., time_column_name
    Parameters
    ----------
    predictions : torch.Tensor  or  List[torch.Tensor]
        Single-target:
            torch.Tensor  [N, look_ahead, n_quantiles]
        Multi-target:
            List[torch.Tensor]  where each entry is [N, look_ahead, n_quantiles]
            — one tensor per target, matching the order of `targets`
    targets : List[str]
        Names of target columns, e.g. ["target"] or ["P_kW", "Q_kVAR"].
    test_df : pd.DataFrame
        Full plant DataFrame with timestamps and actual target values.
        Must cover at least start_idx + N + look_ahead rows.
    time_column_name : str
        Name of the timestamp column (e.g. "start_time_utc").
    quantiles : List[float], optional
        Quantile levels corresponding to the last dim of predictions.
        Defaults to [0.1, 0.5, 0.9].
    start_idx : int, optional
        Row offset in test_df where the query set begins.
        Typically --> int(n_total_windows * train_ratio).
        Defaults to 0.

    Attributes
    ----------
    result_dict : dict { float → List[pd.DataFrame] }
        Keys   : quantile levels
        Values : list of DataFrames, one per sliding window
    """

    def __init__(
        self,
        predictions:      Union[torch.Tensor, List[torch.Tensor]],
        targets:          List[str],
        test_df:          pd.DataFrame,
        time_column_name: str,
        quantiles:        List[float] = None,
        start_idx:        int         = 0,
    ):
        self.targets          = list(targets)
        self.test_df          = test_df.reset_index(drop=True)
        self.time_column_name = time_column_name
        self.quantiles        = quantiles or [0.1, 0.5, 0.9]
        self.result_dict: dict = {}

        # ── Normalise input to List[np.ndarray] ───────────────────
        # Single-target tensor   → wrap in list
        # Multi-target list      → convert each to numpy
        # Result: pred_list[t]   → [N, look_ahead, n_quantiles]
        if isinstance(predictions, torch.Tensor):
            pred_list = [predictions.detach().cpu().numpy()]
        else:
            pred_list = [p.detach().cpu().numpy() for p in predictions]

        # ── Validation ────────────────────────────────────────────
        assert len(pred_list) == len(self.targets), (
            f"Got {len(pred_list)} prediction arrays "
            f"but {len(self.targets)} targets: {self.targets}"
        )
        N, look_ahead, n_q = pred_list[0].shape
        assert n_q == len(self.quantiles), (
            f"Model output has {n_q} quantile dims but "
            f"{len(self.quantiles)} quantiles specified: {self.quantiles}"
        )

        # ── Build result_dict ─────────────────────────────────────
        for q_idx, q_val in enumerate(self.quantiles):
            self.result_dict[q_val] = self._build_sample_list(
                pred_list, q_idx, N, look_ahead, start_idx
            )

    def _build_sample_list(
        self,
        pred_list:  List[np.ndarray],   # one array per target [N, T, Q]
        q_idx:      int,                # which quantile to extract
        N:          int,                # number of sliding windows
        look_ahead: int,                # forecast horizon in steps
        start_idx:  int,                # row offset into test_df
    ) -> list:
        """
        Build list of DataFrames for one quantile level.

        For each window i (0 … N-1):
            - Extract timestamps: test_df.iloc[start_idx+i : start_idx+i+look_ahead]
            - For each target: build a "/Predicted" column from its pred array
            - Append timestamp column last (matches colleague's convention)

        Returns
        -------
        List[pd.DataFrame], length N
            columns: [target_1/Predicted, ..., target_n/Predicted, time_column_name]
            index  : aligned with test_df via iloc
        """
        sample_list = []

        for i in range(N):
            row_start = start_idx + i
            row_end = row_start + look_ahead
            # iloc — safe regardless of whether index is int, datetime etc.
            time_slice = self.test_df[self.time_column_name].iloc[
                row_start:row_end]

            # ── Clip to available rows (last window may be shorter) ───
            actual_len = len(time_slice)
            if actual_len < look_ahead:  # ← was: if actual_len == 0
                continue  # skip partial AND empty windows

            dfs = []

            # One column per target (multi-target support from colleague)
            for t_idx, target in enumerate(self.targets):
                pred_df = pd.DataFrame(
                    {target + "/Predicted":
                    pred_list[t_idx][i, :actual_len, q_idx]},
                    index=time_slice.index,
                )
                dfs.append(pred_df)

            # Timestamp column last — matches transform_data() convention
            dfs.append(time_slice)
            sample_list.append(pd.concat(dfs, axis=1))

        return sample_list