
# ''' Plots measured vs. predicted active power per plant (full window) as HTML.'''
# import os
# import numpy as np
# import torch
# import logging

# log = logging.getLogger("meta-netze")


# class PredictionPlotEvaluator:
#     def __init__(self, pred_cache, plant_id_cache, output_dir,
#                  quantiles=None, flatten="sequential"):
#         """
#         pred_cache      : list of (preds, y_test) per test task
#                           preds  -> [n_win, look_ahead, out(=n_quantiles)]
#                           y_test -> [n_win, look_ahead, 1]
#         plant_id_cache  : parallel list, one plant id per task
#         flatten         : "sequential" -> concatenate every window end-to-end
#                           "first_step"  -> take only step 0 of each window
#         """
#         self.pred_cache = pred_cache
#         self.plant_ids  = plant_id_cache
#         self.output_dir = output_dir
#         self.quantiles  = quantiles or [0.5]
#         self.flatten    = flatten
#         os.makedirs(self.output_dir, exist_ok=True)

#     def _median_idx(self):
#         q = np.asarray(self.quantiles)
#         return int(np.argmin(np.abs(q - 0.5)))

#     def _group_by_plant(self):
#         qi = self._median_idx()
#         grouped = {}
#         n = min(len(self.pred_cache), len(self.plant_ids)) \
#             if self.plant_ids else len(self.pred_cache)

#         for t in range(len(self.pred_cache)):
#             preds, y_true = self.pred_cache[t]            
#             pid = self._extract_pid(
#                 self.plant_ids[t] if (self.plant_ids and t < len(self.plant_ids)) else None,
#                 t,
#             )

#             preds  = preds.detach().cpu().numpy()   # [n_win, look_ahead, out]
#             y_true = y_true.detach().cpu().numpy()   # [n_win, look_ahead, 1]

#             if self.flatten == "first_step":
#                 yhat = preds[:, 0, qi]
#                 yt   = y_true[:, 0, 0]
#             else:  # sequential: stitch windows end-to-end
#                 yhat = preds[:, :, qi].reshape(-1)
#                 yt   = y_true[:, :, 0].reshape(-1)

#             g = grouped.setdefault(pid, {"y_true": [], "y_hat": []})
#             g["y_true"].append(yt)
#             g["y_hat"].append(yhat)

#         for pid, g in grouped.items():
#             g["y_true"] = np.concatenate(g["y_true"])
#             g["y_hat"]  = np.concatenate(g["y_hat"])
#         return grouped

#     @staticmethod
#     def _extract_pid(raw, t):
#         """Return a hashable, human-readable plant id from a cache entry."""
#         if isinstance(raw, dict):
#             for key in ("plant_id", "id", "TFTTest_id", "name"):
#                 if key in raw and raw[key] is not None:
#                     return str(raw[key])
#             # fallback: first value in the dict
#             return str(next(iter(raw.values()))) if raw else f"task{t}"
#         if raw is None:
#             return f"task{t}"
#         return str(raw)
#     def run(self):
#         try:
#             import plotly.graph_objects as go
#         except ImportError:
#             log.error("plotly required for HTML plots.")
#             return

#         grouped = self._group_by_plant()
#         summary = ["plant_id,n_points,MAE"]
#         for pid, g in grouped.items():
#             y_true, y_hat = g["y_true"], g["y_hat"]
#             mae = float(np.nanmean(np.abs(y_true - y_hat)))

#             fig = go.Figure()
#             fig.add_trace(go.Scatter(y=y_true, mode="lines",
#                                      name="measured", line=dict(width=1)))
#             fig.add_trace(go.Scatter(y=y_hat, mode="lines",
#                                      name="predicted", line=dict(width=1)))
#             fig.update_layout(
#                 title=f"Plant {pid} — measured vs. predicted | MAE={mae:.4f}",
#                 xaxis_title="time step", yaxis_title="active power (kW)",
#                 template="plotly_white", hovermode="x unified",
#             )
#             fig.write_html(os.path.join(
#                 self.output_dir, f"plant_{pid}_pred_vs_true.html"))
#             summary.append(f"{pid},{len(y_true)},{mae:.6f}")

#         with open(os.path.join(self.output_dir, "prediction_summary.csv"), "w") as f:
#             f.write("\n".join(summary) + "\n")
#         log.info(f"[PredictionPlotEvaluator] {len(grouped)} plant plots → {self.output_dir}")
#         print(f"✅ {len(grouped)} plant prediction plots saved to {self.output_dir}")


# ''' Plots measured vs. predicted active power per plant (full window) as HTML.'''
# import os
# import numpy as np
# import torch
# import logging

# log = logging.getLogger("meta-netze")


# class PredictionPlotEvaluator:
#     def __init__(self, pred_cache, plant_id_cache, output_dir,
#                  quantiles=None, flatten="sequential",
#                  select="all", n_each=5):
#         """
#         pred_cache      : list of (preds, y_test) per test task
#                           preds  -> [n_win, look_ahead, out(=n_quantiles)]
#                           y_test -> [n_win, look_ahead, 1]
#         plant_id_cache  : parallel list, one plant id per task
#         flatten         : "sequential" -> concatenate every window end-to-end
#                           "first_step"  -> take only step 0 of each window
#         select          : "all" -> plot every plant
#                           "cherry" -> only top/middle/bottom n_each by MAE
#         n_each          : number of plants per group when select="cherry"
#         """
#         self.pred_cache = pred_cache
#         self.plant_ids  = plant_id_cache
#         self.output_dir = output_dir
#         self.quantiles  = quantiles or [0.5]
#         self.flatten    = flatten
#         self.select     = select
#         self.n_each     = n_each
#         os.makedirs(self.output_dir, exist_ok=True)

#     def _median_idx(self):
#         q = np.asarray(self.quantiles)
#         return int(np.argmin(np.abs(q - 0.5)))

#     def _group_by_plant(self):
#         qi = self._median_idx()
#         grouped = {}
#         # n = min(len(self.pred_cache), len(self.plant_ids)) \
#         #     if self.plant_ids else len(self.pred_cache)
#         n_q = len(self.quantiles)

#         for t in range(len(self.pred_cache)):
#             preds, y_true = self.pred_cache[t]
#             pid = self._extract_pid(
#                 self.plant_ids[t] if (self.plant_ids and t < len(self.plant_ids)) else None,
#                 t,
#             )

#             preds  = preds.detach().cpu().numpy()   # [n_win, look_ahead, out]
#             y_true = y_true.detach().cpu().numpy()   # [n_win, look_ahead, 1]

#             if self.flatten == "first_step":
#                 yhat = preds[:, 0, qi]
#                 yt   = y_true[:, 0, 0]
#                 q_all = {i: preds[:, 0, i]           for i in range(n_q)}
#             else:  # sequential: stitch windows end-to-end
#                 yhat = preds[:, :, qi].reshape(-1)
#                 yt   = y_true[:, :, 0].reshape(-1)
#                 q_all = {i: preds[:, :, i].reshape(-1) for i in range(n_q)}

#             # g = grouped.setdefault(pid, {"y_true": [], "y_hat": []})
#             g = grouped.setdefault(pid, {
#                 "y_true": [], "y_hat": [],
#                 "q_all": {i: [] for i in range(n_q)},
#             })
#             g["y_true"].append(yt)
#             g["y_hat"].append(yhat)
#             for i in range(n_q):
#                 g["q_all"][i].append(q_all[i])

#         for pid, g in grouped.items():
#             g["y_true"] = np.concatenate(g["y_true"])
#             g["y_hat"]  = np.concatenate(g["y_hat"])
#             g["q_all"]  = {i: np.concatenate(v) for i, v in g["q_all"].items()}
#             g["mae"]    = float(np.nanmean(np.abs(g["y_true"] - g["y_hat"])))
#         return grouped

#     def _pick(self, grouped):
#         """Return {pid: label}. label='all' unless cherry-picking."""
#         if self.select != "cherry":
#             return {pid: "all" for pid in grouped}

#         ranked = sorted(grouped.items(), key=lambda kv: kv[1]["mae"])  # best -> worst
#         n = len(ranked)
#         picks = {}
#         for pid, _ in ranked[:self.n_each]:                 # best (lowest MAE)
#             picks[pid] = "best"
#         start = max(0, n // 2 - self.n_each // 2)           # middle
#         for pid, _ in ranked[start:start + self.n_each]:
#             picks.setdefault(pid, "middle")
#         for pid, _ in ranked[-self.n_each:]:                # worst (highest MAE)
#             picks.setdefault(pid, "worst")
#         return picks

#     @staticmethod
#     def _extract_pid(raw, t):
#         """Return a hashable, human-readable plant id from a cache entry."""
#         if isinstance(raw, dict):
#             for key in ("plant_id", "id", "TFTTest_id", "name"):
#                 if key in raw and raw[key] is not None:
#                     return str(raw[key])
#             # fallback: first value in the dict
#             return str(next(iter(raw.values()))) if raw else f"task{t}"
#         if raw is None:
#             return f"task{t}"
#         return str(raw)

#     def run(self):
#         try:
#             import plotly.graph_objects as go
#         except ImportError:
#             log.error("plotly required for HTML plots.")
#             return

#         grouped = self._group_by_plant()
#         picks   = self._pick(grouped)

#         summary = ["group,plant_id,n_points,MAE"]
#         for pid, label in sorted(picks.items(), key=lambda kv: (kv[1], grouped[kv[0]]["mae"])):
#             g = grouped[pid]
#             y_true, y_hat, mae = g["y_true"], g["y_hat"], g["mae"]

#             fig = go.Figure()
#             fig.add_trace(go.Scatter(y=y_true, mode="lines",
#                                      name="measured", line=dict(width=1)))
#             fig.add_trace(go.Scatter(y=y_hat, mode="lines",
#                                      name="predicted", line=dict(width=1)))
#             fig.update_layout(
#                 title=f"[{label}] Plant {pid} — measured vs. predicted | MAE={mae:.4f}",
#                 xaxis_title="time step", yaxis_title="active power (kW)",
#                 template="plotly_white", hovermode="x unified",
#             )
#             fname = f"{label}_plant_{pid}_pred_vs_true.html" if label != "all" \
#                     else f"plant_{pid}_pred_vs_true.html"
#             fig.write_html(os.path.join(self.output_dir, fname))
#             summary.append(f"{label},{pid},{len(y_true)},{mae:.6f}")

#         with open(os.path.join(self.output_dir, "prediction_summary.csv"), "w") as f:
#             f.write("\n".join(summary) + "\n")
#         log.info(f"[PredictionPlotEvaluator] {len(picks)} plant plots → {self.output_dir}")
#         print(f"✅ {len(picks)} plant prediction plots ({self.select}) saved to {self.output_dir}")


''' Plots measured vs. predicted active power per plant (full window) as HTML.'''

# import os
# import numpy as np
# import torch
# import pandas as pd
# import logging

# log = logging.getLogger("meta-netze")


# class PredictionPlotEvaluator:
#     def __init__(self, pred_cache, plant_id_cache, output_dir,
#                  quantiles=None, flatten="sequential",
#                  select="all", n_each=5, time_cache=None,
#                  support_cache=None):
#         self.pred_cache = pred_cache
#         self.plant_ids  = plant_id_cache
#         self.output_dir = output_dir
#         self.quantiles  = quantiles or [0.5]
#         self.flatten    = flatten
#         self.select     = select
#         self.n_each     = n_each
#         self.time_cache = time_cache
#         self.support_cache = support_cache
#         os.makedirs(self.output_dir, exist_ok=True)

#     def _median_idx(self):
#         q = np.asarray(self.quantiles)
#         return int(np.argmin(np.abs(q - 0.5)))

#     def _group_by_plant(self):
#         qi  = self._median_idx()
#         n_q = len(self.quantiles)
#         grouped = {}

#         for t in range(len(self.pred_cache)):
#             preds, y_true = self.pred_cache[t]
#             pid = self._extract_pid(
#                 self.plant_ids[t] if (self.plant_ids and t < len(self.plant_ids)) else None,
#                 t,
#             )

#             preds  = preds.detach().cpu().numpy()    # [n_win, look_ahead, n_q]
#             y_true = y_true.detach().cpu().numpy()   # [n_win, look_ahead, 1]

#             if self.flatten == "first_step":
#                 yhat  = preds[:, 0, qi]
#                 yt    = y_true[:, 0, 0]
#                 q_all = {i: preds[:, 0, i]           for i in range(n_q)}
#             else:
#                 yhat  = preds[:, :, qi].reshape(-1)
#                 yt    = y_true[:, :, 0].reshape(-1)
#                 q_all = {i: preds[:, :, i].reshape(-1) for i in range(n_q)}

#             g = grouped.setdefault(pid, {
#                 "y_true": [], "y_hat": [],
#                 "q_all": {i: [] for i in range(n_q)},
#                 "times": [],
#                 "support_y": [],
#             })
#             g["y_true"].append(yt)
#             if self.support_cache and t < len(self.support_cache):
#                 _, y_sup = self.support_cache[t]
#                 if hasattr(y_sup, "detach"):
#                     y_sup = y_sup.detach().cpu().numpy()
#                 y_sup = np.asarray(y_sup)
#                 g["support_y"].append(y_sup[:, :, 0].reshape(-1))
#             g["y_hat"].append(yhat)
#             for i in range(n_q):
#                             # ← ADD block
#                 if self.time_cache and t < len(self.time_cache):
#                     tc = np.asarray(self.time_cache[t])       # [n_win, look_ahead]
#                     if self.flatten == "first_step":
#                         g["times"].append(tc[:, 0] if tc.ndim == 2 else tc)
#                     else:
#                         g["times"].append(tc.reshape(-1) if tc.ndim == 2 else tc)
#                 g["q_all"][i].append(q_all[i])

#         for pid, g in grouped.items():
#             g["y_true"] = np.concatenate(g["y_true"])
#             g["y_hat"]  = np.concatenate(g["y_hat"])
#             g["q_all"]  = {i: np.concatenate(v) for i, v in g["q_all"].items()}
#             g["mae"]    = float(np.nanmean(np.abs(g["y_true"] - g["y_hat"])))
#             # g["times"] = (pd.to_datetime(np.concatenate(g["times"]))
#             #               if g["times"] else None)        # ← ADD
#             valid_times = [np.atleast_1d(t) for t in g["times"] if t is not None]
#             g["times"] = (pd.to_datetime(np.concatenate(valid_times))
#               if valid_times else pd.DatetimeIndex([]))
#             g["support_y"] = (np.concatenate(g["support_y"])
#                   if g["support_y"] else np.array([]))
#         return grouped

#     def _pick(self, grouped):
#         if self.select != "cherry":
#             return {pid: "all" for pid in grouped}
#         ranked = sorted(grouped.items(), key=lambda kv: kv[1]["mae"])
#         n = len(ranked)
#         picks = {}
#         for pid, _ in ranked[:self.n_each]:
#             picks[pid] = "best"
#         start = max(0, n // 2 - self.n_each // 2)
#         for pid, _ in ranked[start:start + self.n_each]:
#             picks.setdefault(pid, "middle")
#         for pid, _ in ranked[-self.n_each:]:
#             picks.setdefault(pid, "worst")
#         return picks

#     @staticmethod
#     def _extract_pid(raw, t):
#         if isinstance(raw, dict):
#             for key in ("plant_id", "id", "TFTTest_id", "name"):
#                 if key in raw and raw[key] is not None:
#                     return str(raw[key])
#             return str(next(iter(raw.values()))) if raw else f"task{t}"
#         if raw is None:
#             return f"task{t}"
#         return str(raw)

#     def _add_quantile_band(self, fig, g, go):
#         """Add shaded q_lo–q_hi band + dashed boundary lines."""
#         n_q = len(self.quantiles)
#         if n_q < 2:
#             return  # nothing to shade with a single quantile

#         x    = list(range(len(g["y_hat"])))
#         q_lo = g["q_all"][0]          # lowest quantile  (e.g. 0.1)
#         q_hi = g["q_all"][n_q - 1]   # highest quantile (e.g. 0.9)

#         lo_label = f"q{self.quantiles[0]:.2f}"
#         hi_label = f"q{self.quantiles[-1]:.2f}"

#         # ── invisible upper boundary (needed for fill='tonexty') ──────
#         fig.add_trace(go.Scatter(
#             x=x, y=q_hi,
#             mode="lines",
#             line=dict(color="rgba(99,110,250,0)"),
#             showlegend=False,
#             hoverinfo="skip",
#         ))
#         # ── shaded area from q_lo up to q_hi ─────────────────────────
#         fig.add_trace(go.Scatter(
#             x=x, y=q_lo,
#             mode="lines",
#             fill="tonexty",
#             fillcolor="rgba(99,110,250,0.15)",
#             line=dict(color="rgba(99,110,250,0)"),
#             name=f"{lo_label}–{hi_label} interval",
#             hoverinfo="skip",
#         ))
#         # ── visible dashed boundary lines ────────────────────────────
#         for arr, label in [(q_lo, lo_label), (q_hi, hi_label)]:
#             fig.add_trace(go.Scatter(
#                 x=x, y=arr,
#                 mode="lines",
#                 name=label,
#                 line=dict(width=1, dash="dot", color="rgba(99,110,250,0.6)"),
#             ))

#     def run(self):
#         try:
#             import plotly.graph_objects as go
#         except ImportError:
#             log.error("plotly required for HTML plots.")
#             return

#         grouped = self._group_by_plant()
#         picks   = self._pick(grouped)

#         summary = ["group,plant_id,n_points,MAE"]
#         for pid, label in sorted(picks.items(),
#                                   key=lambda kv: (kv[1], grouped[kv[0]]["mae"])):
#             g = grouped[pid]
#             y_true, y_hat, mae = g["y_true"], g["y_hat"], g["mae"]
#             x = (g["times"] if (self.time_cache and g["times"] is not None)
#                  else list(range(len(y_hat))))

#             fig = go.Figure()

#             # ── quantile band (rendered first → sits behind lines) ────
#             self._add_quantile_band(fig, g, go)

#             # ── median prediction ─────────────────────────────────────
#             fig.add_trace(go.Scatter(
#                 x=x, y=y_hat, mode="lines",
#                 name="q0.50 (median)",
#                 line=dict(width=1.5, color="royalblue"),
#             ))
#             # ── ground truth ──────────────────────────────────────────
#             fig.add_trace(go.Scatter(
#                 x=x, y=y_true, mode="lines",
#                 name="measured",
#                 line=dict(width=1, color="black"),
#             ))

#             fig.update_layout(
#                 title=f"[{label}] Plant {pid} — measured vs. predicted | MAE={mae:.4f}",
#                 xaxis_title="date / time" if self.time_cache else "time step",
#                 xaxis_tickformat="%b %Y", 
#                 yaxis_title="active power (normalised)",
#                 template="plotly_white",
#                 hovermode="x unified",
#             )

#             fname = (f"{label}_plant_{pid}_pred_vs_true.html"
#                      if label != "all"
#                      else f"plant_{pid}_pred_vs_true.html")
#             fig.write_html(os.path.join(self.output_dir, fname))
#             # ── support set plot ──────────────────────────────────────────
#             if self.support_cache and len(g["support_y"]) > 0:
#                 fig_sup = go.Figure()
#                 fig_sup.add_trace(go.Scatter(
#                     y=g["support_y"], mode="lines",
#                     name="support (measured)",
#                     line=dict(width=1, color="darkorange"),
#                 ))
#                 fig_sup.update_layout(
#                     title=f"[{label}] Plant {pid} — support set (measured only)",
#                     xaxis_title="time step",
#                     yaxis_title="active power (normalised)",
#                     template="plotly_white",
#                     hovermode="x unified",
#                 )
#                 fname_sup = (f"{label}_plant_{pid}_support.html"
#                             if label != "all"
#                             else f"plant_{pid}_support.html")
#                 fig_sup.write_html(os.path.join(self.output_dir, fname_sup))
                        
#             summary.append(f"{label},{pid},{len(y_true)},{mae:.6f}")

#         with open(os.path.join(self.output_dir, "prediction_summary.csv"), "w") as f:
#             f.write("\n".join(summary) + "\n")
#         log.info(f"[PredictionPlotEvaluator] {len(picks)} plant plots → {self.output_dir}")
#         print(f"✅ {len(picks)} plant prediction plots ({self.select}) saved to {self.output_dir}")

''' Plots measured vs. predicted active power per plant (full window) as HTML.'''
import os
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("meta-netze")


class PredictionPlotEvaluator:
    def __init__(self, pred_cache, plant_id_cache, output_dir,
                 quantiles=None, flatten="sequential",
                 select="all", n_each=5,
                 time_cache=None,
                 support_cache=None,
                 support_time_cache=None):          # ← for future random sampling
        self.pred_cache         = pred_cache
        self.plant_ids          = plant_id_cache
        self.output_dir         = output_dir
        self.quantiles          = quantiles or [0.5]
        self.flatten            = flatten   # sequential, first_step or average
        self.select             = select
        self.n_each             = n_each
        self.time_cache         = time_cache
        self.support_cache      = support_cache
        self.support_time_cache = support_time_cache
        os.makedirs(self.output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    def _median_idx(self):
        q = np.asarray(self.quantiles)
        return int(np.argmin(np.abs(q - 0.5)))

    # ──────────────────────────────────────────────────────────────────
    # def _group_by_plant(self):
    #     qi  = self._median_idx()
    #     n_q = len(self.quantiles)
    #     grouped = {}

    #     for t in range(len(self.pred_cache)):
    #         preds, y_true = self.pred_cache[t]
    #         pid = self._extract_pid(
    #             self.plant_ids[t] if (self.plant_ids and t < len(self.plant_ids)) else None,
    #             t,
    #         )

    #         preds  = preds.detach().cpu().numpy()    # [n_win, look_ahead, n_q]
    #         y_true = y_true.detach().cpu().numpy()   # [n_win, look_ahead, 1]

    #         if self.flatten == "first_step":
    #             yhat  = preds[:, 0, qi]
    #             yt    = y_true[:, 0, 0]
    #             q_all = {i: preds[:, 0, i]             for i in range(n_q)}
    #         else:
    #             yhat  = preds[:, :, qi].reshape(-1)
    #             yt    = y_true[:, :, 0].reshape(-1)
    #             q_all = {i: preds[:, :, i].reshape(-1) for i in range(n_q)}

    #         g = grouped.setdefault(pid, {
    #             "y_true":        [],
    #             "y_hat":         [],
    #             "q_all":         {i: [] for i in range(n_q)},
    #             "times":         [],
    #             "support_y":     [],
    #             "support_times": [],
    #             # ── used only when flatten="average" ──────────────
    #             "y_hat_2d":      [],          # list of [n_win, look_ahead]
    #             "y_true_2d":     [],
    #             "q_all_2d":      {i: [] for i in range(n_q)},
    #             "times_2d":      [],          # list of [n_win, look_ahead] timestamps
    #         })

    #         # ── per-task flattening ───────────────────────────────
    #         if self.flatten == "first_step":
    #             yhat  = preds[:, 0, qi]
    #             yt    = y_true[:, 0, 0]
    #             q_all = {i: preds[:, 0, i] for i in range(n_q)}

    #             g["y_true"].append(yt)
    #             g["y_hat"].append(yhat)
    #             for i in range(n_q):
    #                 g["q_all"][i].append(q_all[i])

    #             if self.time_cache and t < len(self.time_cache):
    #                 tc = np.asarray(self.time_cache[t])
    #                 g["times"].append(tc[:, 0] if tc.ndim == 2 else np.atleast_1d(tc))

    #         elif self.flatten == "average":
    #             # keep 2-D; overlaps resolved at consolidation
    #             g["y_hat_2d"].append(preds[:, :, qi])        # [n_win, look_ahead]
    #             g["y_true_2d"].append(y_true[:, :, 0])       # [n_win, look_ahead]
    #             for i in range(n_q):
    #                 g["q_all_2d"][i].append(preds[:, :, i])  # [n_win, look_ahead]

    #             if self.time_cache and t < len(self.time_cache):
    #                 tc = np.asarray(self.time_cache[t])       # [n_win, look_ahead]
    #                 if tc.ndim == 2:
    #                     g["times_2d"].append(tc)
    #                 else:
    #                     log.warning(f"[{pid}] task {t}: flatten='average' needs "
    #                                 f"time_cache shape [n_win, look_ahead], "
    #                                 f"got {tc.shape} → timestamps skipped for this task")

    #         else:  # "sequential" (default)
    #             yhat  = preds[:, :, qi].reshape(-1)
    #             yt    = y_true[:, :, 0].reshape(-1)
    #             q_all = {i: preds[:, :, i].reshape(-1) for i in range(n_q)}

    #             g["y_true"].append(yt)
    #             g["y_hat"].append(yhat)
    #             for i in range(n_q):
    #                 g["q_all"][i].append(q_all[i])

    #             if self.time_cache and t < len(self.time_cache):
    #                 tc = np.asarray(self.time_cache[t])
    #                 g["times"].append(tc.reshape(-1) if tc.ndim == 2 else np.atleast_1d(tc))

    #         # ── support values (unchanged) ────────────────────────
    #         if self.support_cache and t < len(self.support_cache):
    #             _, y_sup = self.support_cache[t]
    #             if hasattr(y_sup, "detach"):
    #                 y_sup = y_sup.detach().cpu().numpy()
    #             y_sup = np.asarray(y_sup)
    #             g["support_y"].append(y_sup[:, :, 0].reshape(-1))

    #         # ── support timestamps (unchanged) ────────────────────
    #         # if self.support_time_cache and t < len(self.support_time_cache):
    #         #     ts = np.asarray(self.support_time_cache[t])
    #         #     g["support_times"].append(ts.reshape(-1) if ts.ndim == 2 else np.atleast_1d(ts))
            
    #         # ── support timestamps (sliced to look_ahead) ─────────
    #         if self.support_time_cache and t < len(self.support_time_cache):
    #             ts = np.asarray(self.support_time_cache[t])

    #             if ts.ndim == 2 and self.support_cache and t < len(self.support_cache):
    #                 _, y_sup_ref = self.support_cache[t]
    #                 y_sup_ref    = np.asarray(y_sup_ref.detach().cpu()
    #                                         if hasattr(y_sup_ref, "detach") else y_sup_ref)
    #                 look_ahead   = y_sup_ref.shape[1]      # [n_win, look_ahead, 1]

    #                 if ts.shape[1] != look_ahead:
    #                     ts = ts[:, -look_ahead:]           # nur der Target-Teil des Fensters

    #                 g["support_times"].append(ts.reshape(-1))
    #             else:
    #                 g["support_times"].append(ts.reshape(-1) if ts.ndim == 2 else np.atleast_1d(ts))

    #     # ── consolidate ───────────────────────────────────────────
    #     for pid, g in grouped.items():

    #         if self.flatten == "average" and g["y_hat_2d"]:
    #             y_hat_all  = np.concatenate(g["y_hat_2d"])   # [total_n_win, look_ahead]
    #             y_true_all = np.concatenate(g["y_true_2d"])  # [total_n_win, look_ahead]

    #             # all tasks must have contributed timestamps for averaging to work
    #             all_times_present = (
    #                 len(g["times_2d"]) == len(g["y_hat_2d"])
    #                 and len(g["times_2d"]) > 0
    #             )

    #             if all_times_present:
    #                 times_all  = np.concatenate(g["times_2d"])    # [total_n_win, look_ahead]
    #                 times_flat = pd.to_datetime(times_all.reshape(-1))
    #                 yhat_flat  = y_hat_all.reshape(-1)
    #                 ytrue_flat = y_true_all.reshape(-1)

    #                 df = pd.DataFrame({
    #                     "t":     times_flat,
    #                     "yhat":  yhat_flat,
    #                     "ytrue": ytrue_flat,
    #                 })
    #                 for i in range(n_q):
    #                     q_arr      = np.concatenate(g["q_all_2d"][i])  # [total_n_win, look_ahead]
    #                     df[f"q{i}"] = q_arr.reshape(-1)

    #                 df = df.groupby("t").mean().sort_index()   # average overlapping windows

    #                 g["y_hat"]  = df["yhat"].values
    #                 g["y_true"] = df["ytrue"].values
    #                 g["times"]  = pd.DatetimeIndex(df.index)
    #                 g["q_all"]  = {i: df[f"q{i}"].values for i in range(n_q)}

    #             else:
    #                 # no usable timestamps → fall back to sequential reshape
    #                 log.warning(f"[{pid}] flatten='average' but timestamps incomplete "
    #                             f"→ falling back to sequential")
    #                 g["y_hat"]  = y_hat_all.reshape(-1)
    #                 g["y_true"] = y_true_all.reshape(-1)
    #                 g["q_all"]  = {i: np.concatenate(g["q_all_2d"][i]).reshape(-1)
    #                                for i in range(n_q)}
    #                 g["times"]  = pd.DatetimeIndex([])

    #         else:
    #             # sequential / first_step  (original logic, untouched)
    #             g["y_true"] = np.concatenate(g["y_true"])
    #             g["y_hat"]  = np.concatenate(g["y_hat"])
    #             g["q_all"]  = {i: np.concatenate(v) for i, v in g["q_all"].items()}

    #             valid_qt = [np.atleast_1d(t) for t in g["times"] if t is not None]
    #             g["times"] = (pd.to_datetime(np.concatenate(valid_qt))
    #                           if valid_qt else pd.DatetimeIndex([]))

    #         g["mae"] = float(np.nanmean(np.abs(g["y_true"] - g["y_hat"])))

    #         # # ── support consolidation (unchanged) ─────────────────
    #         # g["support_y"] = (np.concatenate(g["support_y"])
    #         #                   if g["support_y"] else np.array([]))

    #         # valid_st = [np.atleast_1d(t) for t in g["support_times"] if t is not None]
    #         # g["support_times"] = (pd.to_datetime(np.concatenate(valid_st))
    #         #                       if valid_st else pd.DatetimeIndex([]))
            
    #         # ── support consolidation ─────────────────────────────
    #         # valid_st = [np.atleast_1d(t) for t in g["support_times"] if t is not None]
    #         # if g["support_y"] and valid_st:
    #         #     sup_series = pd.Series(
    #         #         np.concatenate(g["support_y"]),
    #         #         index=pd.to_datetime(np.concatenate(valid_st)),
    #         #     ).sort_index()
    #         #     sup_series = sup_series[~sup_series.index.duplicated(keep="first")]
    #         #     g["support_y"]     = sup_series.values
    #         #     g["support_times"] = sup_series.index
    #         # else:
    #         #     g["support_y"]     = np.concatenate(g["support_y"]) if g["support_y"] else np.array([])
    #         #     g["support_times"] = pd.to_datetime(np.concatenate(valid_st)) if valid_st else pd.DatetimeIndex([])

    #         # ── support consolidation ─────────────────────────────
    #         valid_st = [np.atleast_1d(t) for t in g["support_times"] if t is not None]

    #         if g["support_y"] and valid_st:
    #             sup_vals = np.concatenate(g["support_y"])
    #             sup_idx  = pd.to_datetime(np.concatenate(valid_st))

    #             if len(sup_vals) != len(sup_idx):
    #                 log.warning(f"[{pid}] support length mismatch: "
    #                             f"values={len(sup_vals)} vs times={len(sup_idx)} "
    #                             f"→ dropping support timestamps")
    #                 g["support_y"]     = sup_vals
    #                 g["support_times"] = pd.DatetimeIndex([])   # run() → Integer-x-Achse
    #             else:
    #                 sup_series = pd.Series(sup_vals, index=sup_idx).sort_index()
    #                 sup_series = sup_series[~sup_series.index.duplicated(keep="first")]
    #                 g["support_y"]     = sup_series.values
    #                 g["support_times"] = sup_series.index
    #         else:
    #             g["support_y"]     = np.concatenate(g["support_y"]) if g["support_y"] else np.array([])
    #             g["support_times"] = (pd.to_datetime(np.concatenate(valid_st))
    #                                 if valid_st else pd.DatetimeIndex([]))

    #         # ── debug ─────────────────────────────────────────────
    #         # print(f"[DEBUG] {pid:30s} | "
    #         #       f"y_hat={len(g['y_hat']):6d}  times={len(g['times']):6d}  "
    #         #       f"support_y={len(g['support_y']):6d}  sup_times={len(g['support_times']):6d}")
    #         # if len(g["times"]) > 0:
    #         #     print(f"         sample times → {g['times'][0]}  …  {g['times'][-1]}")
    #         # else:
    #         #     print(f"         ⚠ NO timestamps → will use integer x-axis")

    #     return grouped

        #     g["y_true"].append(yt)
        #     g["y_hat"].append(yhat)

        #     # quantile loop — NO other appends inside here
        #     for i in range(n_q):
        #         g["q_all"][i].append(q_all[i])

        #     # ── query timestamps (OUTSIDE quantile loop) ──────────────
        #     if self.time_cache and t < len(self.time_cache):
        #         tc = np.asarray(self.time_cache[t])            # [n_win, look_ahead]
        #         if self.flatten == "first_step":
        #             g["times"].append(tc[:, 0] if tc.ndim == 2 else np.atleast_1d(tc))
        #         else:
        #             g["times"].append(tc.reshape(-1) if tc.ndim == 2 else np.atleast_1d(tc))

        #     # ── support values ────────────────────────────────────────
        #     if self.support_cache and t < len(self.support_cache):
        #         _, y_sup = self.support_cache[t]
        #         if hasattr(y_sup, "detach"):
        #             y_sup = y_sup.detach().cpu().numpy()
        #         y_sup = np.asarray(y_sup)
        #         g["support_y"].append(y_sup[:, :, 0].reshape(-1))

        #     # ── support timestamps ────────────────────────────────────
        #     if self.support_time_cache and t < len(self.support_time_cache):
        #         ts = np.asarray(self.support_time_cache[t])
        #         g["support_times"].append(ts.reshape(-1) if ts.ndim == 2 else np.atleast_1d(ts))

        # # ── consolidate ───────────────────────────────────────────────
        # for pid, g in grouped.items():
        #     g["y_true"] = np.concatenate(g["y_true"])
        #     g["y_hat"]  = np.concatenate(g["y_hat"])
        #     g["q_all"]  = {i: np.concatenate(v) for i, v in g["q_all"].items()}
        #     g["mae"]    = float(np.nanmean(np.abs(g["y_true"] - g["y_hat"])))

        #     valid_qt = [np.atleast_1d(t) for t in g["times"] if t is not None]
        #     g["times"] = (pd.to_datetime(np.concatenate(valid_qt))
        #                   if valid_qt else pd.DatetimeIndex([]))

        #     g["support_y"] = (np.concatenate(g["support_y"])
        #                       if g["support_y"] else np.array([]))

        #     valid_st = [np.atleast_1d(t) for t in g["support_times"] if t is not None]
        #     g["support_times"] = (pd.to_datetime(np.concatenate(valid_st))
        #                           if valid_st else pd.DatetimeIndex([]))

        #     # ── debug ──────────────────────────────────────────────────
        #     print(f"[DEBUG] {pid:30s} | "
        #           f"y_hat={len(g['y_hat']):6d}  times={len(g['times']):6d}  "
        #           f"support_y={len(g['support_y']):6d}  sup_times={len(g['support_times']):6d}")
        #     if len(g["times"]) > 0:
        #         print(f"         sample times → {g['times'][0]}  …  {g['times'][-1]}")
        #     else:
        #         print(f"         ⚠ NO timestamps → will use integer x-axis")

        # return grouped

    def _group_by_plant(self):
        qi  = self._median_idx()
        n_q = len(self.quantiles)
        grouped = {}

        for t in range(len(self.pred_cache)):
            preds, y_true = self.pred_cache[t]
            pid = self._extract_pid(
                self.plant_ids[t] if (self.plant_ids and t < len(self.plant_ids)) else None,
                t,
            )

            preds  = preds.detach().cpu().numpy()    # [n_win, look_ahead, n_q]
            y_true = y_true.detach().cpu().numpy()   # [n_win, look_ahead, 1]

            g = grouped.setdefault(pid, {
                "y_true":        [],
                "y_hat":         [],
                "q_all":         {i: [] for i in range(n_q)},
                "times":         [],
                "support_y":     [],
                "support_times": [],
                "y_hat_2d":      [],
                "y_true_2d":     [],
                "q_all_2d":      {i: [] for i in range(n_q)},
                "times_2d":      [],
            })

            # ── per-task flattening ───────────────────────────────────────
            if self.flatten == "first_step":
                yhat  = preds[:, 0, qi]
                yt    = y_true[:, 0, 0]
                q_all = {i: preds[:, 0, i] for i in range(n_q)}

                g["y_true"].append(yt)
                g["y_hat"].append(yhat)
                for i in range(n_q):
                    g["q_all"][i].append(q_all[i])

                if self.time_cache and t < len(self.time_cache):
                    tc = np.asarray(self.time_cache[t])
                    g["times"].append(tc[:, 0] if tc.ndim == 2 else np.atleast_1d(tc))

            elif self.flatten == "average":
                g["y_hat_2d"].append(preds[:, :, qi])
                g["y_true_2d"].append(y_true[:, :, 0])
                for i in range(n_q):
                    g["q_all_2d"][i].append(preds[:, :, i])

                if self.time_cache and t < len(self.time_cache):
                    tc = np.asarray(self.time_cache[t])
                    if tc.ndim == 2:
                        g["times_2d"].append(tc)
                    else:
                        log.warning(
                            f"[{pid}] task {t}: flatten='average' needs "
                            f"time_cache shape [n_win, look_ahead], "
                            f"got {tc.shape} → timestamps skipped for this task"
                        )

            else:  # "sequential" (default)
                yhat  = preds[:, :, qi].reshape(-1)
                yt    = y_true[:, :, 0].reshape(-1)
                q_all = {i: preds[:, :, i].reshape(-1) for i in range(n_q)}

                g["y_true"].append(yt)
                g["y_hat"].append(yhat)
                for i in range(n_q):
                    g["q_all"][i].append(q_all[i])

                if self.time_cache and t < len(self.time_cache):
                    tc = np.asarray(self.time_cache[t])
                    g["times"].append(tc.reshape(-1) if tc.ndim == 2 else np.atleast_1d(tc))

            # ── support values + timestamps (always collected together) ───
            if self.support_cache and t < len(self.support_cache):
                _, y_sup = self.support_cache[t]
                if hasattr(y_sup, "detach"):
                    y_sup = y_sup.detach().cpu().numpy()
                y_sup     = np.asarray(y_sup)           # [n_win, look_ahead, 1]
                y_sup_flat = y_sup[:, :, 0].reshape(-1) # [n_win * look_ahead]

                ts_flat = None
                if self.support_time_cache and t < len(self.support_time_cache):
                    ts_raw = self.support_time_cache[t]
                    if ts_raw is not None:
                        ts = np.asarray(ts_raw)
                        if ts.ndim == 2:
                            look_ahead = y_sup.shape[1]
                            if ts.shape[1] != look_ahead:
                                ts = ts[:, -look_ahead:]    # slice to target window only
                            ts_flat = ts.reshape(-1)
                        elif ts.ndim == 1:
                            ts_flat = ts
                        else:
                            ts_flat = np.atleast_1d(ts)

                        # per-task length guard — drop ts if mismatched
                        if ts_flat is not None and len(ts_flat) != len(y_sup_flat):
                            log.warning(
                                f"[{pid}] task {t}: support_times len={len(ts_flat)} "
                                f"!= support_y len={len(y_sup_flat)} "
                                f"→ timestamps dropped for this task"
                            )
                            ts_flat = None

                g["support_y"].append(y_sup_flat)
                if ts_flat is not None:
                    g["support_times"].append(ts_flat)   # only appended when in sync

        # ── consolidate ───────────────────────────────────────────────────
        for pid, g in grouped.items():

            if self.flatten == "average" and g["y_hat_2d"]:
                y_hat_all  = np.concatenate(g["y_hat_2d"])   # [total_n_win, look_ahead]
                y_true_all = np.concatenate(g["y_true_2d"])  # [total_n_win, look_ahead]

                all_times_present = (
                    len(g["times_2d"]) == len(g["y_hat_2d"])
                    and len(g["times_2d"]) > 0
                )

                if all_times_present:
                    times_all  = np.concatenate(g["times_2d"])   # [total_n_win, look_ahead]
                    times_flat = pd.to_datetime(times_all.reshape(-1))
                    yhat_flat  = y_hat_all.reshape(-1)
                    ytrue_flat = y_true_all.reshape(-1)

                    df = pd.DataFrame({
                        "t":     times_flat,
                        "yhat":  yhat_flat,
                        "ytrue": ytrue_flat,
                    })
                    for i in range(n_q):
                        q_arr       = np.concatenate(g["q_all_2d"][i])
                        df[f"q{i}"] = q_arr.reshape(-1)

                    df = df.groupby("t").mean().sort_index()

                    g["y_hat"]  = df["yhat"].values
                    g["y_true"] = df["ytrue"].values
                    g["times"]  = pd.DatetimeIndex(df.index)
                    g["q_all"]  = {i: df[f"q{i}"].values for i in range(n_q)}

                else:
                    log.warning(
                        f"[{pid}] flatten='average' but timestamps incomplete "
                        f"→ falling back to sequential"
                    )
                    g["y_hat"]  = y_hat_all.reshape(-1)
                    g["y_true"] = y_true_all.reshape(-1)
                    g["q_all"]  = {
                        i: np.concatenate(g["q_all_2d"][i]).reshape(-1)
                        for i in range(n_q)
                    }
                    g["times"]  = pd.DatetimeIndex([])

            else:  # sequential / first_step
                g["y_true"] = np.concatenate(g["y_true"])
                g["y_hat"]  = np.concatenate(g["y_hat"])
                g["q_all"]  = {i: np.concatenate(v) for i, v in g["q_all"].items()}

                valid_qt = [np.atleast_1d(t) for t in g["times"] if t is not None]
                g["times"] = (
                    pd.to_datetime(np.concatenate(valid_qt))
                    if valid_qt else pd.DatetimeIndex([])
                )

            g["mae"] = float(np.nanmean(np.abs(g["y_true"] - g["y_hat"])))

            # ── support consolidation ─────────────────────────────────────
            g["support_y"] = np.concatenate(g["support_y"]) if g["support_y"] else np.array([])

            valid_st = [np.atleast_1d(t) for t in g["support_times"] if t is not None]
            if valid_st:
                sup_idx = pd.to_datetime(np.concatenate(valid_st))
                if len(sup_idx) == len(g["support_y"]):
                    sup_series = (
                        pd.Series(g["support_y"], index=sup_idx)
                        .sort_index()
                    )
                    sup_series         = sup_series[~sup_series.index.duplicated(keep="first")]
                    g["support_y"]     = sup_series.values
                    g["support_times"] = sup_series.index
                else:
                    log.warning(
                        f"[{pid}] support length mismatch after concat: "
                        f"values={len(g['support_y'])} vs times={len(sup_idx)} "
                        f"→ using integer x-axis for support plot"
                    )
                    g["support_times"] = pd.DatetimeIndex([])
            else:
                g["support_times"] = pd.DatetimeIndex([])

        return grouped

    # ──────────────────────────────────────────────────────────────────
    def _pick(self, grouped):
        if self.select != "cherry":
            return {pid: "all" for pid in grouped}
        ranked = sorted(grouped.items(), key=lambda kv: kv[1]["mae"])
        n = len(ranked)
        picks = {}
        for pid, _ in ranked[:self.n_each]:
            picks[pid] = "best"
        start = max(0, n // 2 - self.n_each // 2)
        for pid, _ in ranked[start:start + self.n_each]:
            picks.setdefault(pid, "middle")
        for pid, _ in ranked[-self.n_each:]:
            picks.setdefault(pid, "worst")
        return picks

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_pid(raw, t):
        if isinstance(raw, dict):
            for key in ("plant_id", "id", "TFTTest_id", "name"):
                if key in raw and raw[key] is not None:
                    return str(raw[key])
            return str(next(iter(raw.values()))) if raw else f"task{t}"
        if raw is None:
            return f"task{t}"
        return str(raw)

    # ──────────────────────────────────────────────────────────────────
    def _add_quantile_band(self, fig, g, go, x):
        """Shaded q_lo–q_hi band. x must match the main traces."""
        n_q = len(self.quantiles)
        if n_q < 2:
            return

        q_lo     = g["q_all"][0]
        q_hi     = g["q_all"][n_q - 1]
        lo_label = f"q{self.quantiles[0]:.2f}"
        hi_label = f"q{self.quantiles[-1]:.2f}"

        # invisible upper boundary needed for fill='tonexty'
        fig.add_trace(go.Scatter(
            x=x, y=q_hi, mode="lines",
            line=dict(color="rgba(99,110,250,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=q_lo, mode="lines",
            fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(99,110,250,0)"),
            name=f"{lo_label}–{hi_label} interval",
            hoverinfo="skip",
        ))
        for arr, lbl in [(q_lo, lo_label), (q_hi, hi_label)]:
            fig.add_trace(go.Scatter(
                x=x, y=arr, mode="lines", name=lbl,
                line=dict(width=1, dash="dot", color="rgba(99,110,250,0.6)"),
            ))

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_layout(title, use_times, yaxis_title="active power (normalised)"):
        kw = dict(
            title=title,
            xaxis_title="Time" if use_times else "Time step",
            yaxis_title=yaxis_title,
            template="plotly_white",
            hovermode="x unified",
        )
        if use_times:
            kw["xaxis_tickformat"] = "%d %b %Y"
        return kw

    # ──────────────────────────────────────────────────────────────────
    def run(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.error("plotly required for HTML plots.")
            return

        grouped = self._group_by_plant()
        picks   = self._pick(grouped)

        has_times   = sum(1 for g in grouped.values() if len(g["times"]) > 0)
        has_support = sum(1 for g in grouped.values() if len(g["support_y"]) > 0)
        print(f"[DEBUG] Plants with query timestamps : {has_times}/{len(grouped)}")
        print(f"[DEBUG] Plants with support data     : {has_support}/{len(grouped)}")

        summary = ["group,plant_id,n_points,MAE"]
        for pid, label in sorted(picks.items(),
                                  key=lambda kv: (kv[1], grouped[kv[0]]["mae"])):
            g = grouped[pid]
            y_true, y_hat, mae = g["y_true"], g["y_hat"], g["mae"]

            # use timestamps only when length matches
            use_times = len(g["times"]) == len(y_hat)
            x = list(g["times"]) if use_times else list(range(len(y_hat)))
            if self.time_cache and not use_times:
                print(f"[DEBUG] ⚠ {pid}: times={len(g['times'])} vs y_hat={len(y_hat)} "
                      f"→ integer x-axis")

            # ── query plot ────────────────────────────────────────────
            fig = go.Figure()
            self._add_quantile_band(fig, g, go, x)      # x passed in
            fig.add_trace(go.Scatter(
                x=x, y=y_hat, mode="lines",
                name="q0.50 (median)",
                line=dict(width=1.5, color="royalblue"),
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_true, mode="lines",
                name="measured",
                line=dict(width=1, color="black"),
            ))
            fig.update_layout(**self._make_layout(
                f"[{label}] Plant {pid} — query: measured vs. predicted | MAE={mae:.4f}",
                use_times,
            ))
            fname = (f"{label}_plant_{pid}_pred_vs_true.html"
                     if label != "all" else f"plant_{pid}_pred_vs_true.html")
            fig.write_html(os.path.join(self.output_dir, fname))

            # ── support set plot ──────────────────────────────────────
            if self.support_cache:
                if len(g["support_y"]) == 0:
                    print(f"[DEBUG] ⚠ {pid}: support_cache supplied but support_y is empty")
                else:
                    use_st = len(g["support_times"]) == len(g["support_y"])
                    x_sup  = list(g["support_times"]) if use_st else list(range(len(g["support_y"])))

                    fig_sup = go.Figure()
                    fig_sup.add_trace(go.Scatter(
                        x=x_sup, y=g["support_y"], mode="lines",
                        name="support (measured)",
                        line=dict(width=1, color="darkorange"),
                    ))
                    fig_sup.update_layout(**self._make_layout(
                        f"[{label}] Plant {pid} — support set (measured only)",
                        use_st,
                    ))
                    fname_sup = (f"{label}_plant_{pid}_support.html"
                                 if label != "all" else f"plant_{pid}_support.html")
                    fig_sup.write_html(os.path.join(self.output_dir, fname_sup))

            summary.append(f"{label},{pid},{len(y_true)},{mae:.6f}")

        with open(os.path.join(self.output_dir, "prediction_summary.csv"), "w") as f:
            f.write("\n".join(summary) + "\n")
        log.info(f"[PredictionPlotEvaluator] {len(picks)} plant plots → {self.output_dir}")
        print(f"✅ {len(picks)} plant prediction plots ({self.select}) saved to {self.output_dir}")