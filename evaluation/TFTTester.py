"""
Compute and plot errors for TFT predictions.
Reuse of the original TFTTester — only pytorch-forecasting imports removed.
"""

import os
import copy
import statistics
import numpy as np
import pandas as pd
# import plotly.express as px
import matplotlib.pyplot as plt
from typing import List
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from evaluation.meta_prediction import MetaPrediction

class TFTTester:
    """
    Computes and saves error plots for a MetaPrediction object.
    Usage
    -----
    pred   = MetaPrediction(predictions, targets, test_df, ...)
    tester = TFTTester(save_path="results/tft/", pred=pred)
    """

    def __init__(self, save_path: str, pred: MetaPrediction, fast_mode: bool = True):
        self.targets          = pred.targets
        self.test_df          = pred.test_df
        self.save_path        = save_path.rstrip("/") + "/"
        self.time_column_name = pred.time_column_name
        self.error_methods    = {
            "mae":   self.mae,
            "mape":  self.mape,
            "mse":   self.mse,
            "smape": self.smape,
            "bias":  self.bias,
        }
        self.pred = pred
        self._actual_cache = {}

        os.makedirs(self.save_path, exist_ok=True)
        # self.plot_predictions()
        if not fast_mode:
            self.plot_predictions()  # slow: per-sample Plotly PDFs
        # Extract a representative plant_id from plant_info
        plant_info = getattr(pred, 'plant_info', [])
        if plant_info and plant_info[0].get("plant_id"):
            plant_id = plant_info[0]["plant_id"]
        else:
            plant_id = "plant"
        
        self.rated_power = self._resolve_rated_power(plant_id)
        self.plot_matplotlib_summary(plant_id=plant_id)

    # ──────────────────────────────────────────────────────────────
    #  Main plotting entry point
    # ──────────────────────────────────────────────────────────────

    def plot_predictions(self):
        # ── Use only quantiles that exist in result_dict ──────────────
        available = list(self.pred.result_dict.keys())
        mae_pred = self.pred.result_dict[0.5]

        # ── Clip time indices to actual window length ─────────────
        window_len = len(mae_pred[0]) if mae_pred else 96

        # Now clips to valid range for any window length
        raw_times = np.linspace(0, window_len - 1, 4, dtype=int).tolist()
        valid_times = [t for t in raw_times if t < window_len]
        if not valid_times:
            valid_times = [window_len - 1]   # at least last step

        # ── Build symmetric quantile pairs (e.g. 0.05/0.95, 0.1/0.9, 0.25/0.75)
        quantile_pairs = []
        for q in sorted(available):
            complement = round(1.0 - q, 4)
            if q < 0.5 and complement in available:
                quantile_pairs.append((q, complement))

        # ── Aggregated error curves over all samples ───────────────
        self.plot_aggr_error(mae_pred, "mae")
        self.plot_aggr_error(mae_pred, "mape")
        self.plot_aggr_error(mae_pred, "smape")
        self.plot_aggr_error(mae_pred, "mse")
        self.plot_aggr_error(mae_pred, "bias")
        self.plot_medae(mae_pred)
        self.plot_r2(mae_pred)

        # ── Sample-level plots at specific forecast horizons ───────
        self.plot_error(mae_pred, valid_times[0],  error="mae", 
                        aggr=True, name="day_aggr")

        # ── Quantile interval plots for each pair ─────────────────
        for q_lo, q_hi in quantile_pairs:
            quantile_list = [q_lo, q_hi]
            quantile_sample_list = [self.pred.result_dict[q] for q in quantile_list]

            self.plot_quantile(
                quantile_sample_list, quantile_list, valid_times[0],
                f"day_aggr_{q_lo},{q_hi}_time:{valid_times[0]}",
                aggr=True,
            )

            for time in valid_times:
                self.plot_quantile(
                    quantile_sample_list, quantile_list, time,
                    f"{q_lo},{q_hi}_time:{time}",
                )

        # ── Per-horizon MAE (independent of quantiles) ────────────
        for time in valid_times:
            self.plot_error(
                mae_pred, time, error="mae", name=f"mae_time:{time}")

    # ──────────────────────────────────────────────────────────────
    #  Core plot helper
    # ──────────────────────────────────────────────────────────────

    def plot_series(
        self,
        data_series_list: list,
        time: pd.Series,
        name: str  = "plot",
        info: str  = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 4))

        for s in data_series_list:
            ax.plot(time.values, s.values, label=s.name, linewidth=1.2)

        ax.set_xlabel(time.name)
        ax.set_ylabel(
            next((t for t in self.targets if t in name), "Value")
        )
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(data_series_list), 4), fontsize=8,
        )
        ax.grid(True, alpha=0.3)

        if info:
            ax.text(
                0.05, 0.95, info, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        safe_name = name.replace("/", "_")
        plt.savefig(self.save_path + f"{safe_name}.png", dpi=100)
        plt.close()
        # data_series_list.append(time)
        # df        = pd.concat(data_series_list, axis=1)
        # plot_name = next(
        #     (t for t in self.targets if t in name), "plot"
        # )
        # fig = px.line(
        #     df, x=time.name,
        #     y=df.columns.drop(time.name),
        #     title=plot_name,
        # )
        # fig.update_layout(
        #     legend=dict(
        #         orientation="h", yanchor="top", y=-0.2,
        #         xanchor="center", x=0.5,
        #     ),
        #     margin=dict(b=150),
        # )
        # fig.add_annotation(
        #     x=0.05, y=0.95, xref="paper", yref="paper", text=info
        # )
        # safe_name = name.replace("/", "_")
        # fig.write_image(self.save_path + f"{safe_name}.pdf")
        # fig = None
        # import gc; gc.collect()
        # Consider using HTML type of images --> FASTER
        # try:
        #     fig.write_image(path + ".pdf")
        # except Exception:
        #     fig.write_html(path + ".html")

    # ──────────────────────────────────────────────────────────────
    #  Actual values helper
    # ──────────────────────────────────────────────────────────────

    def create_actual(self, time: pd.Series) -> pd.DataFrame:
        actual = self.test_df.copy()
        actual = actual.loc[actual[self.time_column_name].isin(time)]
        cols   = self.targets + [self.time_column_name]
        actual = actual[cols].rename(
            columns={t: t + "/Actual" for t in self.targets}
        )
        return actual

    # ──────────────────────────────────────────────────────────────
    #  Quantile interval plot
    # ──────────────────────────────────────────────────────────────

    def plot_quantile(
        self,
        quantile_sample_list,
        quantile_list: list,
        time:          int,
        name:          str  = "",
        aggr:          bool = False,
    ):
        result    = []
        for qs in quantile_sample_list:
            result.append(self.get_time(qs, time, aggr))

        timesteps = result[0][0][self.time_column_name]

        for target in self.targets:
            list_to_plot = []
            for index, dfs in enumerate(result):
                predicted       = dfs[0][target + "/Predicted"].copy()
                predicted.name  = (
                    target + "/Predicted"
                    + f"_{quantile_list[index]}_time:{time}"
                )
                list_to_plot.append(predicted)

            actual      = result[0][1][target + "/Actual"].copy()
            actual.name = target + "/Actual"
            list_to_plot.append(actual)

            conf = ""
            if len(quantile_list) == 2:
                lo  = list_to_plot[0].tolist()
                hi  = list_to_plot[1].tolist()
                act = list_to_plot[2].tolist()
                n_in = sum(
                    1 for i in range(len(act))
                    if lo[i] <= act[i] <= hi[i]
                )
                conf = f"{round(n_in / len(act) * 100)}% in confint."
            else:
                conf = ""

            self.plot_series(
                list_to_plot, timesteps,
                name=f"{name}_{target}", info=conf,
            )

    # ──────────────────────────────────────────────────────────────
    #  Error plots
    # ──────────────────────────────────────────────────────────────

    def plot_error(
        self,
        mae_pred:  list,
        time:      int,
        aggr:      bool = False,
        error:     str  = "",
        plot_act:  bool = True,
        plot_pred: bool = True,
        name:      str  = "",
    ):
        predicted, actual = self.get_time(mae_pred, time, aggr)
        timesteps         = predicted[self.time_column_name]
        list_to_plot      = []

        if error in self.error_methods:
            list_to_plot.append(self.error_methods[error](predicted, actual))
        if plot_act:
            list_to_plot.append(actual)
        if plot_pred:
            list_to_plot.append(predicted)

        for target in self.targets:
            target_plot = [df.filter(like=target).squeeze() for df in list_to_plot]
            for s in target_plot:
                s.index = timesteps.index
            self.plot_series(
                target_plot, timesteps,
                name=f"{name}_{error}_{target}_time:{time}",
            )

    def plot_aggr_error(self, mae_pred: list, error: str = ""):
        actual_list  = [self.create_actual(s[self.time_column_name]) for s in mae_pred]
        error_list   = [
            self.error_methods[error](mae_pred[i], actual_list[i])
            for i in range(len(actual_list))
        ]
        timesteps        = pd.Series(range(len(mae_pred[0])))
        timesteps.name   = self.time_column_name

        for df in error_list:
            df.reset_index(inplace=True, drop=True)

        result       = error_list[0].copy()
        numeric_cols = result.select_dtypes(include="number").columns
        for df in error_list[1:]:
            result[numeric_cols] = result[numeric_cols] + df[numeric_cols]
        result[numeric_cols] = result[numeric_cols] / len(error_list)

        for target in self.targets:
            self.plot_series(
                [result.filter(like=target).squeeze()],
                timesteps, name=f"{error}_{target}",
            )

    def plot_medae(self, mae_pred: list):
        actual_list = [self.create_actual(s[self.time_column_name]) for s in mae_pred]
        error_list  = []
        timesteps   = pd.Series(range(len(mae_pred[0])))
        timesteps.name = self.time_column_name

        for ind, sample in enumerate(mae_pred):
            er = self.mae(sample, actual_list[ind])
            er.reset_index(inplace=True, drop=True)
            error_list.append(er)

        # ── Filter to full-length windows only ────────────────────────
        expected_len = len(error_list[0])
        original_count = len(error_list)
        error_list = [e for e in error_list if len(e) == expected_len]
        dropped = original_count - len(error_list)
        if dropped:
            print(f"[TFTTester.plot_medae] Dropped {dropped}/{original_count} "
                  f"partial windows (expected length={expected_len})")
        if not error_list:
            print("[TFTTester.plot_medae] No full-length windows — skipping.")
            return
        
        # ── Vectorized median ─────────────────────────────────────
        target_cols = [t + "MAE" for t in self.targets]
        stacked = np.stack([e[target_cols].values for e in error_list])  # [N, L, T]
        medians = np.median(stacked, axis=0)  # [L, T]

        for t_idx, target in enumerate(self.targets):
            series = pd.Series(medians[:, t_idx], name=target + "MedAE")
            self.plot_series([series], timesteps, name=f"MedAE_{target}")

        # result = error_list[0].copy()
        # for i, x in enumerate(error_list[0].index):
        #     for target in self.targets:
        #         col = target + "MAE"
        #         vals = [s[col].iloc[i] for s in error_list]
        #         result.at[x, col] = statistics.median(vals)

        # result = result.rename(
        #     columns={t + "MAE": t + "MedAE" for t in self.targets}
        # )
        # for target in self.targets:
        #     self.plot_series(
        #         [result[target + "MedAE"]], timesteps,
        #         name=f"MedAE_{target}",
        #     )

    # def plot_r2(self, mae_pred: list):
    #     mae_pred    = copy.deepcopy(mae_pred)
    #     actual_list = [self.create_actual(s[self.time_column_name])
    #                    for s in mae_pred]
    #     timesteps   = pd.Series(range(len(mae_pred[0])))
    #     timesteps.name = self.time_column_name

    #     for pred in mae_pred:
    #         pred.reset_index(inplace=True, drop=True)
    #     for act in actual_list:
    #         act.reset_index(inplace=True, drop=True)

    #     # ── Filter to full-length windows only ────────────────────────
    #     expected_len = len(mae_pred[0])
    #     original_count = len(mae_pred)
    #     pairs = [(p, a) for p, a in zip(mae_pred, actual_list)
    #              if len(p) == expected_len and len(a) == expected_len]
    #     dropped = original_count - len(pairs)
    #     if dropped:
    #         print(f"[TFTTester.plot_r2] Dropped {dropped}/{original_count} "
    #               f"partial windows (expected length={expected_len})")
    #     if not pairs:
    #         print("[TFTTester.plot_r2] No full-length windows — skipping.")
    #         return
    #     mae_pred, actual_list = zip(*pairs)

    #     result = (mae_pred[0].copy()
    #               .astype({t + "/Predicted": float for t in self.targets}))
    #     for ind in mae_pred[0].index:
    #         for target in self.targets:
    #             pred_vals = [s[target + "/Predicted"][ind] for s in mae_pred]
    #             act_vals  = [s[target + "/Actual"][ind]    for s in actual_list]
    #             y_bar     = np.mean(act_vals)

    #             ss_res = sum((p - a) ** 2 for p, a in zip(pred_vals, act_vals))
    #             ss_tot = sum((a - y_bar) ** 2 for a in act_vals)

    #             r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    #             result.at[ind, target + "/Predicted"] = r2

    #     result = result.rename(
    #         columns={t + "/Predicted": t + "R2" for t in self.targets}
    #     )
    #     for target in self.targets:
    #         self.plot_series(
    #             [result[target + "R2"]], timesteps,
    #             name=f"R2_{target}",
    #         )
    def plot_r2(self, mae_pred: list):
        actual_list = [self.create_actual(s[self.time_column_name]) for s in mae_pred]
        timesteps = pd.Series(range(len(mae_pred[0])))
        timesteps.name = self.time_column_name

        # Reset indices
        mae_pred_reset = [s.reset_index(drop=True) for s in mae_pred]
        actual_reset = [a.reset_index(drop=True) for a in actual_list]

        # Filter to full-length windows
        expected_len = len(mae_pred_reset[0])
        pairs = [
            (p, a) for p, a in zip(mae_pred_reset, actual_reset)
            if len(p) == expected_len and len(a) == expected_len
        ]
        if not pairs:
            print("[TFTTester.plot_r2] No full-length windows — skipping.")
            return

        preds_reset, actuals_reset = zip(*pairs)

        # ── Vectorized R² ─────────────────────────────────────────
        for t_idx, target in enumerate(self.targets):
            pred_col = target + "/Predicted"
            act_col = target + "/Actual"

            # Stack into [N, L] arrays
            pred_arr = np.stack([s[pred_col].values for s in preds_reset])    # [N, L]
            act_arr = np.stack([s[act_col].values for s in actuals_reset])    # [N, L]

            # R² at each timestep across all windows
            # y_bar[l] = mean of actuals at step l across all windows
            y_bar = act_arr.mean(axis=0)                      # [L]
            ss_res = np.sum((pred_arr - act_arr) ** 2, axis=0)  # [L]
            ss_tot = np.sum((act_arr - y_bar[None, :]) ** 2, axis=0)  # [L]

            r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)  # [L]

            series = pd.Series(r2, name=target + "R2")
            self.plot_series([series], timesteps, name=f"R2_{target}")
    # ──────────────────────────────────────────────────────────────
    #  Error metric methods
    # ──────────────────────────────────────────────────────────────

    def mae(self, sample: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        result = sample.copy()
        for t in self.targets:
            result[t + "/Predicted"] = np.abs(
                sample[t + "/Predicted"] - actual[t + "/Actual"]
            )
        return result.rename(columns={t + "/Predicted": t + "MAE" for t in self.targets})

    def mse(self, sample: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        result = sample.copy()
        for t in self.targets:
            result[t + "/Predicted"] = (
                sample[t + "/Predicted"] - actual[t + "/Actual"]
            ) ** 2
        return result.rename(columns={t + "/Predicted": t + "MSE" for t in self.targets})

    def mape(self, sample: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        result = sample.copy()
        for t in self.targets:
            result[t + "/Predicted"] = np.abs(
                sample[t + "/Predicted"] - actual[t + "/Actual"]
            ) / np.abs(actual[t + "/Actual"]).replace(0, np.nan)
        return result.rename(columns={t + "/Predicted": t + "MAPE" for t in self.targets})

    def smape(self, sample: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        result = sample.copy()
        for t in self.targets:
            result[t + "/Predicted"] = np.abs(
                sample[t + "/Predicted"] - actual[t + "/Actual"]
            ) / (np.abs(sample[t + "/Predicted"] + actual[t + "/Actual"]) / 2).replace(0, np.nan)
        return result.rename(columns={t + "/Predicted": t + "SMAPE" for t in self.targets})

    def bias(self, sample: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        result = sample.copy()
        for t in self.targets:
            result[t + "/Predicted"] = (
                sample[t + "/Predicted"] - actual[t + "/Actual"]
            )
        return result.rename(columns={t + "/Predicted": t + "Bias" for t in self.targets})

    # ──────────────────────────────────────────────────────────────
    #  get_time
    # ──────────────────────────────────────────────────────────────

    def get_time(
        self,
        sample_list: list,
        time:        int,
        aggr:        bool = False,
    ) -> tuple[DataFrame, DataFrame]:
        if aggr:
            rows   = [
                s.apply(lambda col: col.sum() / len(col)
                        if is_numeric_dtype(col) else col.iloc[0])
                for s in sample_list
            ]
            result = pd.DataFrame(rows)
            act_rows = [
                self.create_actual(s[self.time_column_name]).apply(
                    lambda col: col.sum() / len(col)
                    if is_numeric_dtype(col) else col.iloc[0]
                )
                for s in sample_list
            ]
            actual = pd.DataFrame(act_rows)
        else:
            rows   = [s.iloc[time] for s in sample_list]
            result = pd.DataFrame(rows)
            actual = self.create_actual(result[self.time_column_name])

        return result, actual
    

        # ──────────────────────────────────────────────────────────────
    #  Bridge: result_dict → numpy arrays
    # ──────────────────────────────────────────────────────────────

    def _extract_arrays(self) -> tuple:
        """
        Extract predictions and actuals from result_dict[0.5] as numpy arrays.

        Returns
        -------
        preds_np   : [N, look_ahead, n_targets]
        actuals_np : [N, look_ahead, n_targets]

        Only full-length windows are included (same guard as plot_r2/plot_medae).
        """
        mae_pred     = self.pred.result_dict[0.5]
        expected_len = len(mae_pred[0])

        preds_list   = []
        actuals_list = []

        for sample_df in mae_pred:
            if len(sample_df) != expected_len:
                continue                           # skip partial windows

            actual_df = self.create_actual(sample_df[self.time_column_name])
            if len(actual_df) != expected_len:
                continue

            pred_row   = np.stack(
                [sample_df[t + "/Predicted"].values for t in self.targets],
                axis=-1,
            )                                      # [look_ahead, n_targets]
            actual_row = np.stack(
                [actual_df[t + "/Actual"].values for t in self.targets],
                axis=-1,
            )                                      # [look_ahead, n_targets]

            preds_list.append(pred_row)
            actuals_list.append(actual_row)

        if not preds_list:
            return np.array([]), np.array([])

        return np.stack(preds_list), np.stack(actuals_list)  # [N, L, T]

    # ──────────────────────────────────────────────────────────────
    #  Matplotlib summary — entry point
    # ──────────────────────────────────────────────────────────────

    def plot_matplotlib_summary(
        self,
        plant_id:   str = "plant",
        n_samples:  int = 3,
        plot_best:  int = 3,
        plot_worst: int = 3,
    ):
        """
        Matplotlib-based plot suite, mirroring the baseline plotter.

        1. Global metrics saved to txt
        2. Evenly spaced sample windows
        3. Best  MAE windows
        4. Worst MAE windows
        5. Mean MAE per forecast step (horizon error curve)
        """
        preds_np, actuals_np = self._extract_arrays()
        if preds_np.size == 0:
            print("[TFTTester] No full-length windows — skipping matplotlib plots.")
            return

        N = len(preds_np)

        # Per-window MAE: [N]
        per_window_mae = np.mean(np.abs(preds_np - actuals_np), axis=(1, 2))

        self._save_metrics(preds_np, actuals_np, per_window_mae, plant_id)

        spaced  = np.linspace(0, N - 1, min(n_samples, N), dtype=int)
        best    = np.argsort(per_window_mae)[:plot_best]
        worst   = np.argsort(per_window_mae)[-plot_worst:][::-1]

        self._plot_windows(preds_np, actuals_np, spaced, plant_id, tag="sample")
        self._plot_windows(preds_np, actuals_np, best,   plant_id, tag="best")
        self._plot_windows(preds_np, actuals_np, worst,  plant_id, tag="worst")
        self._plot_mae_per_step(preds_np, actuals_np, plant_id)
        self._save_aggregate_metrics(preds_np, actuals_np)

        print(f"[TFTTester] Matplotlib plots saved to: {self.save_path}")

    # ──────────────────────────────────────────────────────────────
    #  Metrics txt
    # ──────────────────────────────────────────────────────────────
   
    def _save_metrics(self, preds_np, actuals_np, per_window_mae, plant_id):
        """
        Save global metrics in both normalized [0,1] and physical (kW) units.

        Denormalization: actual_kW = normalized_value × rated_power_kW
        """
        mae_norm  = float(np.mean(per_window_mae))
        rmse_norm = float(np.sqrt(np.mean((preds_np - actuals_np) ** 2)))

        # ── Denormalization ───────────────────────────────────────
        # from data.dataset_classes.PVTimeSeriesDatasetV2 import PVTimeSeriesDatasetV2
        # self.rated_power = PVTimeSeriesDatasetV2._rated_power.get(plant_id, None)

        # # Also check plant_info attached to MetaPrediction
        # if self.rated_power is None and hasattr(self.pred, 'plant_info'):
        #     # Try to find this plant in plant_info cache
        #     for info in self.pred.plant_info:
        #         if info.get("plant_id") == plant_id:
        #             self.rated_power = info.get("self.rated_power_kW")
        #             break

        has_kw = self.rated_power is not None and self.rated_power > 0

        if has_kw:
            mae_kw  = mae_norm * self.rated_power
            rmse_kw = rmse_norm * self.rated_power
        else:
            mae_kw  = None
            rmse_kw = None

        # ── MAPE: daytime only ────────────────────────────────────
        mask = actuals_np > 0.01 * actuals_np.max()
        mape = float(
            np.mean(np.abs(
                (preds_np[mask] - actuals_np[mask]) / actuals_np[mask]
            )) * 100
        ) if mask.any() else float("nan")

        # ── Print ─────────────────────────────────────────────────
        print(f"\n[TFTTester] Global metrics — {plant_id}:")
        print(f"  Rated capacity : {self.rated_power:.1f} kW" if has_kw else
              f"  Rated capacity : UNKNOWN (no Stammdaten entry)")
        print(f"  MAE  (norm)    : {mae_norm:.4f}")
        if has_kw:
            print(f"  MAE  (kW)      : {mae_kw:.2f} kW")
        print(f"  RMSE (norm)    : {rmse_norm:.4f}")
        if has_kw:
            print(f"  RMSE (kW)      : {rmse_kw:.2f} kW")
        print(f"  MAPE (daytime) : {mape:.2f}%")

        # ── Save to file ──────────────────────────────────────────
        path = os.path.join(self.save_path, f"{plant_id}_metrics.txt")
        with open(path, "w") as f:
            f.write(f"Plant              : {plant_id}\n")
            f.write(f"Rated capacity     : "
                    f"{f'{self.rated_power:.1f} kW' if has_kw else 'UNKNOWN'}\n")
            f.write(f"Test windows       : {len(preds_np)}\n")
            f.write(f"─── Normalized [0,1] ───\n")
            f.write(f"  MAE              : {mae_norm:.6f}\n")
            f.write(f"  RMSE             : {rmse_norm:.6f}\n")
            if has_kw:
                f.write(f"─── Physical (kW) ───\n")
                f.write(f"  MAE              : {mae_kw:.2f} kW\n")
                f.write(f"  RMSE             : {rmse_kw:.2f} kW\n")
                f.write(f"  nMAE             : {mae_kw / self.rated_power * 100:.2f}% "
                        f"of rated capacity\n")
            f.write(f"─── Relative ───\n")
            f.write(f"  MAPE (daytime)   : {mape:.2f}%\n")
            f.write(f"─── Per-window ───\n")
            f.write(
                f"  Best  window MAE : {per_window_mae.min():.6f}"
                f"  (idx={per_window_mae.argmin()})\n"
            )
            f.write(
                f"  Worst window MAE : {per_window_mae.max():.6f}"
                f"  (idx={per_window_mae.argmax()})\n"
            )
    # ──────────────────────────────────────────────────────────────
    #  Per-window prediction plots
    # ──────────────────────────────────────────────────────────────

    def _plot_windows(self, preds_np, actuals_np, indices, plant_id, tag: str):
        """
        Plot predicted vs actual for each window in indices.
        If rated_power is known, plot in kW units. Otherwise normalized.
        """

        has_kw = self.rated_power is not None and self.rated_power > 0
        scale = self.rated_power if has_kw else 1.0
        unit  = "kW" if has_kw else "(normalized)"

        for idx in indices:
            for t_idx, target in enumerate(self.targets):
                pred   = preds_np  [idx, :, t_idx] * scale
                actual = actuals_np[idx, :, t_idx] * scale
                steps  = np.arange(len(pred))
                hours  = steps / 4  # 15-min → hours
                mae    = float(np.mean(np.abs(pred - actual)))

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(hours, actual, label="Actual",
                        color="steelblue", linewidth=1.5)
                ax.plot(hours, pred,   label="Predicted",
                        color="tomato",   linewidth=1.5, linestyle="--")
                ax.set_title(
                    f"{plant_id} — {target} [{tag}] "
                    f"Window {idx}  |  MAE={mae:.2f} {unit}"
                    + (f"  |  Rated: {self.rated_power:.0f} kW" if has_kw else "")
                )
                ax.set_xlabel("Forecast Horizon (hours)")
                ax.set_ylabel(f"Power ({unit})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Mark night (zero actual) for context
                if has_kw:
                    night_mask = actual < 0.01 * self.rated_power
                    for start, end in self._contiguous_regions(night_mask):
                        ax.axvspan(hours[start], hours[min(end, len(hours)-1)],
                                   alpha=0.05, color='gray')

                plt.tight_layout()

                safe_target = target.replace("/", "_")
                path = os.path.join(
                    self.save_path,
                    f"{plant_id}_{safe_target}_{tag}_window{idx}.png",
                )
                plt.savefig(path, dpi=150)
                plt.close()

    @staticmethod
    def _contiguous_regions(mask):
        """Find contiguous True regions in a boolean array."""
        d = np.diff(mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends   = np.where(d == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        return zip(starts, ends)

    # ──────────────────────────────────────────────────────────────
    #  MAE per forecast step
    # ──────────────────────────────────────────────────────────────
    def _plot_mae_per_step(self, preds_np, actuals_np, plant_id):
        """
        Mean MAE at each forecast step. Dual y-axis if rated_power known.
        """
        has_kw = self.rated_power is not None and self.rated_power > 0

        mae_per_step = np.mean(np.abs(preds_np - actuals_np), axis=0)  # [L, T]
        steps = np.arange(mae_per_step.shape[0])
        hours = steps / 4

        for t_idx, target in enumerate(self.targets):
            curve_norm = mae_per_step[:, t_idx]

            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax1.plot(hours, curve_norm, color="darkorange", linewidth=1.5)
            ax1.fill_between(hours, curve_norm, alpha=0.15, color="darkorange")
            ax1.set_xlabel("Forecast Horizon (hours)")
            ax1.set_ylabel("MAE (normalized [0,1])")
            ax1.grid(True, alpha=0.3)

            title = (f"{plant_id} — {target}\n"
                     f"Mean MAE per Forecast Step  ({len(preds_np)} windows)")

            if has_kw:
                # Secondary y-axis in kW
                ax2 = ax1.twinx()
                curve_kw = curve_norm * self.rated_power
                ax2.plot(hours, curve_kw, color="darkorange", linewidth=0)  # invisible
                ax2.set_ylabel(f"MAE (kW)  [rated: {self.rated_power:.0f} kW]")
                ax2.set_ylim(ax1.get_ylim()[0] * self.rated_power,
                             ax1.get_ylim()[1] * self.rated_power)
                title += f"  |  Rated: {self.rated_power:.0f} kW"

            ax1.set_title(title, fontsize=10)
            plt.tight_layout()

            safe_target = target.replace("/", "_")
            path = os.path.join(
                self.save_path,
                f"{plant_id}_{safe_target}_mae_per_step.png",
            )
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  MAE-per-step plot saved: {path}")

    def _save_aggregate_metrics(self, preds_np, actuals_np):
        """
        Save aggregate denormalized metrics across all test tasks.
        Uses per-task plant_info if available.
        """
        from data.dataset_classes.PVTimeSeriesDatasetV2 import PVTimeSeriesDatasetV2

        plant_info = getattr(self.pred, 'plant_info', [])
        if not plant_info:
            return

        # Map each window back to its task's rated power
        # pred_cache has one entry per task; each task contributes some windows
        # For now: use the global approach — check if all test plants share info
        self.rated_powers = [
            info.get("rated_power_kW")
            for info in plant_info
            if info.get("rated_power_kW") is not None
        ]

        if not self.rated_powers:
            return

        # Compute per-plant nMAE (normalized MAE as % of capacity)
        plant_metrics = []
        for info in plant_info:
            pid = info.get("plant_id")
            rp  = info.get("rated_power_kW")
            if pid is None or rp is None or rp <= 0:
                continue
            plant_metrics.append({
                "plant_id": pid,
                "rated_power_kW": rp,
            })

        if plant_metrics:
            path = os.path.join(self.save_path, "aggregate_metrics.txt")
            with open(path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("AGGREGATE TEST METRICS\n")
                f.write("=" * 60 + "\n\n")

                # Overall normalized metrics
                mae_norm = float(np.mean(np.abs(preds_np - actuals_np)))
                rmse_norm = float(np.sqrt(np.mean((preds_np - actuals_np) ** 2)))

                f.write(f"Total test windows : {len(preds_np)}\n")
                f.write(f"Total test plants  : {len(plant_metrics)}\n\n")
                f.write(f"── Normalized [0,1] ──\n")
                f.write(f"  MAE  : {mae_norm:.6f}\n")
                f.write(f"  RMSE : {rmse_norm:.6f}\n\n")

                # Per-plant summary table
                f.write(f"── Per-Plant Summary ──\n")
                f.write(f"{'Plant ID':<15s} {'Rated (kW)':>10s} "
                        f"{'MAE (kW)':>10s} {'nMAE (%)':>10s}\n")
                f.write("-" * 50 + "\n")

                total_mae_kw = []
                for info in plant_metrics:
                    rp = info["rated_power_kW"]
                    mae_kw = mae_norm * rp  # approximate (same norm MAE per plant)
                    nmae = mae_norm * 100
                    total_mae_kw.append(mae_kw)
                    f.write(f"{info['plant_id']:<15s} {rp:>10.1f} "
                            f"{mae_kw:>10.2f} {nmae:>10.2f}\n")

                f.write("-" * 50 + "\n")
                f.write(f"{'MEDIAN':<15s} "
                        f"{np.median([m['rated_power_kW'] for m in plant_metrics]):>10.1f} "
                        f"{np.median(total_mae_kw):>10.2f} "
                        f"{mae_norm * 100:>10.2f}\n")

            print(f"  Aggregate metrics saved: {path}")

    def _resolve_rated_power(self, plant_id: str):
        from data.dataset_classes.PVTimeSeriesDatasetV2 import PVTimeSeriesDatasetV2
        rp = PVTimeSeriesDatasetV2._rated_power.get(plant_id)
        if rp is None and hasattr(self.pred, 'plant_info'):
            for info in self.pred.plant_info:
                if info.get("plant_id") == plant_id:
                    return info.get("rated_power_kW")
        return rp
    
    def create_actual(self, time: pd.Series) -> pd.DataFrame:
        key = tuple(time.values)
        if key in self._actual_cache:
            return self._actual_cache[key]

        actual = self.test_df.loc[
            self.test_df[self.time_column_name].isin(time),
            self.targets + [self.time_column_name]
        ].rename(
            columns={t: t + "/Actual" for t in self.targets}
        )

        self._actual_cache[key] = actual
        return actual