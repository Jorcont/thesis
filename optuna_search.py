# """optuna_search.py"""
# import gc
# import itertools
# import uuid
# import yaml
# import optuna
# import torch
# import pytorch_lightning as pl
# import pandas as pd

# from conf.configs import PipelineParams

# BASE_CONFIG_PATH = "conf/yaml/pv_tft_v3.yaml"
# METRIC_CANDIDATES = ["meta_val_loss_masked", "meta_val_loss_masked_epoch",
#                       "meta_val_loss", "meta_val_loss_epoch"]


# ALGORITHM = "MAML"   # "ANIL" oder "MAML "NoInnerLoop" oder "basline true"

# # ── Vom Supervisor vorgegebenes, EXHAUSTIVES Grid ────────────────────
# GRID_HIDDEN_SIZE = [8, 16, 32, 64, 128]
# GRID_NUM_HEADS = [2, 4, 8]
# GRID_META_ITER = [3, 5, 10]
# FIXED_WINDOW_STRIDE = 12

# # ── Anzahl TPE-Trials PRO Grid-Kombination ───────────────────────────
# INNER_TRIALS_PER_COMBO = 8


# def load_base_config() -> dict:
#     with open(BASE_CONFIG_PATH, "r") as f:
#         return yaml.safe_load(f)


# def get_metric(callback_metrics: dict) -> float:
#     for key in METRIC_CANDIDATES:
#         if key in callback_metrics:
#             val = callback_metrics[key]
#             return float(val.item() if hasattr(val, "item") else val)
#     raise KeyError(
#         f"Keine der erwarteten Metriken gefunden. Verfügbare Keys: "
#         f"{list(callback_metrics.keys())}"
#     )


# # def build_trial_config(trial: optuna.Trial, hidden_size: int,
# #                         num_heads: int, meta_iter: int) -> dict:
# #     cfg = load_base_config()

# #     cfg["name"] = (f"pv_tft_optuna_h{hidden_size}_nh{num_heads}_"
# #                    f"it{meta_iter}_{trial.number}_{uuid.uuid4().hex[:6]}")

# #     # ── Fest vom Supervisor vorgegeben (Grid) ─────────────────────
# #     cfg["model_config"]["config"]["hidden_size"] = hidden_size
# #     cfg["model_config"]["config"]["num_heads"] = num_heads
# #     cfg["meta_training_iter"] = meta_iter
# #     cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE

# #     # ── TPE-Suche: die 5 zu tunenden Parameter ────────────────────
# #     cfg["meta_lr"] = trial.suggest_float("meta_lr", 1e-6, 5e-4, log=True)
# #     cfg["fast_lr"] = trial.suggest_float("fast_lr", 1e-3, 3e-1, log=True)
# #     cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
# #     cfg["model_config"]["config"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
# #     cfg["model_config"]["config"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)

# #     # ── Alles andere bleibt unverändert aus base yaml ─────────────
# #     # adaptation_steps, N_train, theta_support/query, etc. → unangetastet.

# #     # ── Sicherheit: validate_every darf nicht > meta_training_iter sein ──
# #     cfg["validate_every"] = min(cfg.get("validate_every", 5), meta_iter)

# #     return cfg
# # def build_trial_config(trial: optuna.Trial, hidden_size: int,
# #                         num_heads: int, meta_iter: int) -> dict:
# #     cfg = load_base_config()

# #     cfg["name"] = (f"pv_tft_optuna_h{hidden_size}_nh{num_heads}_"
# #                    f"it{meta_iter}_{trial.number}_{uuid.uuid4().hex[:6]}")

# #     # ── WICHTIG: schnelle, normale Validierung statt Sample-Efficiency-Kurve ──
# #     cfg["print_meta_val"] = False

# #     # ── Fest vom Supervisor vorgegeben (Grid) ─────────────────────
# #     cfg["model_config"]["config"]["hidden_size"] = hidden_size
# #     cfg["model_config"]["config"]["num_heads"] = num_heads
# #     cfg["meta_training_iter"] = meta_iter
# #     cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE

# #     # ── TPE-Suche: die 5 zu tunenden Parameter ────────────────────
# #     cfg["meta_lr"] = trial.suggest_float("meta_lr", 1e-6, 5e-4, log=True)
# #     cfg["fast_lr"] = trial.suggest_float("fast_lr", 1e-3, 3e-1, log=True)
# #     cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
# #     cfg["model_config"]["config"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
# #     cfg["model_config"]["config"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)

# #     cfg["validate_every"] = min(cfg.get("validate_every", 5), meta_iter)

# #     # ── Empfehlung: schnellere Suche durch kleinere Batch-Limits ──
# #     cfg["trainer_config"]["limit_train_batches"] = 50
# #     cfg["trainer_config"]["limit_val_batches"] = 50

# #     return cfg

# def build_trial_config(trial: optuna.Trial, hidden_size: int,
#                         num_heads: int, meta_iter: int) -> dict:
#     cfg = load_base_config()

#     cfg["name"] = (f"pv_tft_optuna_h{hidden_size}_nh{num_heads}_"
#                    f"it{meta_iter}_{trial.number}_{uuid.uuid4().hex[:6]}")

#     # print_meta_val NICHT überschreiben -> bleibt True (aus base yaml)
#     # -> state.train=False in validation_step -> korrekter, getesteter Code-Pfad

#     cfg["model_config"]["config"]["hidden_size"] = hidden_size
#     cfg["model_config"]["config"]["num_heads"] = num_heads
#     cfg["meta_training_iter"] = meta_iter
#     cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE

#     cfg["meta_lr"] = trial.suggest_float("meta_lr", 1e-6, 5e-4, log=True)
#     cfg["fast_lr"] = trial.suggest_float("fast_lr", 1e-3, 3e-1, log=True)
#     cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
#     cfg["model_config"]["config"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
#     cfg["model_config"]["config"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)

#     cfg["validate_every"] = min(cfg.get("validate_every", 5), meta_iter)

#     # ── Performance-Kompensation statt print_meta_val zu ändern: ──
#     # Die extended_inner_learning() (Sample-Efficiency-Kurve) bleibt aktiv,
#     # aber wir begrenzen die Anzahl der Val-Batches drastisch, damit
#     # die Suche trotzdem schnell bleibt.
#     cfg["trainer_config"]["limit_train_batches"] = 200
#     cfg["trainer_config"]["limit_val_batches"] = 50   # statt 200

#     return cfg


# def make_objective(hidden_size: int, num_heads: int, meta_iter: int):
#     """Erzeugt eine objective()-Funktion für eine feste Grid-Kombination."""
#     def objective(trial: optuna.Trial) -> float:
#         cfg_dict = build_trial_config(trial, hidden_size, num_heads, meta_iter)

#         tmp_config_path = f"/tmp/optuna_cfg_{cfg_dict['name']}.yaml"
#         with open(tmp_config_path, "w") as f:
#             yaml.safe_dump(cfg_dict, f)

#         pipeline = None
#         config = None
#         try:
#             config = PipelineParams(config_path=tmp_config_path, **cfg_dict)
#             pl.seed_everything(cfg_dict.get("seed", 42), workers=True)

#             pipeline = config.pipeline
#             pipeline.meta_training()

#             metric = get_metric(pipeline.trainer.callback_metrics)

#             if metric != metric:  # NaN
#                 raise optuna.TrialPruned("NaN metric")

#             return metric

#         finally:
#             if pipeline is not None:
#                 del pipeline
#             if config is not None:
#                 del config
#             gc.collect()
#             torch.cuda.empty_cache()

#     return objective


# def main():
#     all_results = []

#     combos = list(itertools.product(GRID_HIDDEN_SIZE, GRID_NUM_HEADS, GRID_META_ITER))
#     total_runs = len(combos) * INNER_TRIALS_PER_COMBO
#     print(f"Gesamt: {len(combos)} Grid-Kombinationen × {INNER_TRIALS_PER_COMBO} TPE-Trials "
#           f"= {total_runs} Läufe insgesamt")

#     for hidden_size, num_heads, meta_iter in combos:
#         study_name = f"grid_h{hidden_size}_nh{num_heads}_it{meta_iter}"
#         study = optuna.create_study(
#             direction="minimize",
#             study_name=study_name,
#             storage="sqlite:///optuna_pv_tft.db",
#             load_if_exists=True,
#         )
#         objective = make_objective(hidden_size, num_heads, meta_iter)
#         study.optimize(objective, n_trials=INNER_TRIALS_PER_COMBO, gc_after_trial=True)

#         all_results.append({
#             "hidden_size": hidden_size,
#             "num_heads": num_heads,
#             "meta_training_iter": meta_iter,
#             "best_value": study.best_value,
#             "best_params": study.best_params,
#         })
#         print(f"[DONE] {study_name} → best_value={study.best_value:.6f}")

#     # ── Globales Beste über alle 45 Grid-Kombinationen ────────────
#     best_overall = min(all_results, key=lambda r: r["best_value"])
#     print("\n" + "=" * 60)
#     print("GLOBAL BEST:")
#     print(best_overall)
#     print("=" * 60)

#     # ── Bestes Config-YAML für finalen Full-Run exportieren ───────
#     best_cfg = load_base_config()
#     best_cfg["model_config"]["config"]["hidden_size"] = best_overall["hidden_size"]
#     best_cfg["model_config"]["config"]["num_heads"] = best_overall["num_heads"]
#     best_cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE
#     best_cfg["meta_lr"] = best_overall["best_params"]["meta_lr"]
#     best_cfg["fast_lr"] = best_overall["best_params"]["fast_lr"]
#     best_cfg["weight_decay"] = best_overall["best_params"]["weight_decay"]
#     best_cfg["model_config"]["config"]["num_layers"] = best_overall["best_params"]["num_layers"]
#     best_cfg["model_config"]["config"]["dropout"] = best_overall["best_params"]["dropout"]
#     best_cfg["meta_training_iter"] = 200  # voller Lauf für finales Training
#     best_cfg["name"] = "pv_timeseries_tft_v2_best"

#     with open("conf/yaml/best_config_anil.yaml", "w") as f:
#         yaml.safe_dump(best_cfg, f)
#     print("→ conf/yaml/best_config_maml.yaml geschrieben.")

#     # with open("conf/yaml/best_config_anil.yaml", "w") as f:
#     #         yaml.safe_dump(best_cfg, f)
#     # print("→ conf/yaml/best_config_maml.yaml geschrieben.")
    
        

#     # ── Alle Ergebnisse als CSV für Analyse/Report ─────────────────
#     pd.DataFrame(all_results).to_csv("optuna_grid_summary.csv", index=False)
#     print("→ optuna_grid_summary.csv geschrieben.")


# if __name__ == "__main__":
#     main()


# """optuna_search.py"""
# import gc
# import itertools
# import uuid
# import yaml
# import optuna
# import torch
# import pytorch_lightning as pl
# import pandas as pd

# from conf.configs import PipelineParams

# BASE_CONFIG_PATH = "conf/yaml/pv_tft_v3.yaml"
# METRIC_CANDIDATES = ["meta_val_loss_masked", "meta_val_loss_masked_epoch",
#                       "meta_val_loss", "meta_val_loss_epoch"]

# # ══════════════════════════════════════════════════════════════════
# # NUR DIESE ZEILE ÄNDERN, UM ZWISCHEN ANIL UND MAML ZU WECHSELN
# # ══════════════════════════════════════════════════════════════════
# ALGORITHM = "ANIL"   # "ANIL" oder "MAML"
# # ══════════════════════════════════════════════════════════════════

# ALGO_TAG = ALGORITHM.lower()

# # ── Vom Supervisor vorgegebenes, EXHAUSTIVES Grid ────────────────────
# GRID_HIDDEN_SIZE = [8, 16, 32, 64, 128]
# GRID_NUM_HEADS = [2, 4, 8]
# GRID_META_ITER = [3, 5, 10]
# FIXED_WINDOW_STRIDE = 12

# # ── Anzahl TPE-Trials PRO Grid-Kombination ───────────────────────────
# INNER_TRIALS_PER_COMBO = 8

# # ── Algorithmus-spezifische Output-Pfade (keine Kollision mehr) ─────
# DB_PATH = f"sqlite:///optuna_pv_tft_{ALGO_TAG}.db"
# CSV_SUMMARY_PATH = f"optuna_grid_summary_{ALGO_TAG}.csv"
# BEST_CONFIG_PATH = f"conf/yaml/best_config_{ALGO_TAG}_baseline.yaml"


# def load_base_config() -> dict:
#     with open(BASE_CONFIG_PATH, "r") as f:
#         return yaml.safe_load(f)


# def get_metric(callback_metrics: dict) -> float:
#     for key in METRIC_CANDIDATES:
#         if key in callback_metrics:
#             val = callback_metrics[key]
#             return float(val.item() if hasattr(val, "item") else val)
#     raise KeyError(
#         f"Keine der erwarteten Metriken gefunden. Verfügbare Keys: "
#         f"{list(callback_metrics.keys())}"
#     )


# def build_trial_config(trial: optuna.Trial, hidden_size: int,
#                         num_heads: int, meta_iter: int) -> dict:
#     cfg = load_base_config()

#     cfg["name"] = (f"pv_tft_optuna_{ALGO_TAG}_h{hidden_size}_nh{num_heads}_"
#                    f"it{meta_iter}_{trial.number}_{uuid.uuid4().hex[:6]}")

#     # ── Algorithmus fest auf den gewählten Wert setzen ────────────
#     cfg["algorithm"] = ALGORITHM

#     # print_meta_val NICHT überschreiben -> bleibt True (aus base yaml)
#     # -> state.train=False in validation_step -> korrekter, getesteter Code-Pfad

#     cfg["model_config"]["config"]["hidden_size"] = hidden_size
#     cfg["model_config"]["config"]["num_heads"] = num_heads
#     cfg["meta_training_iter"] = meta_iter
#     cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE

#     cfg["meta_lr"] = trial.suggest_float("meta_lr", 1e-6, 5e-4, log=True)
#     cfg["fast_lr"] = trial.suggest_float("fast_lr", 1e-3, 3e-1, log=True)
#     cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
#     cfg["model_config"]["config"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
#     cfg["model_config"]["config"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)

#     cfg["validate_every"] = min(cfg.get("validate_every", 5), meta_iter)

#     cfg["trainer_config"]["limit_train_batches"] = 200
#     cfg["trainer_config"]["limit_val_batches"] = 50

#     return cfg


# def make_objective(hidden_size: int, num_heads: int, meta_iter: int):
#     """Erzeugt eine objective()-Funktion für eine feste Grid-Kombination."""
#     def objective(trial: optuna.Trial) -> float:
#         cfg_dict = build_trial_config(trial, hidden_size, num_heads, meta_iter)

#         tmp_config_path = f"/tmp/optuna_cfg_{cfg_dict['name']}.yaml"
#         with open(tmp_config_path, "w") as f:
#             yaml.safe_dump(cfg_dict, f)

#         pipeline = None
#         config = None
#         try:
#             config = PipelineParams(config_path=tmp_config_path, **cfg_dict)
#             pl.seed_everything(cfg_dict.get("seed", 42), workers=True)

#             pipeline = config.pipeline
#             pipeline.meta_training()

#             metric = get_metric(pipeline.trainer.callback_metrics)

#             if metric != metric:  # NaN
#                 raise optuna.TrialPruned("NaN metric")

#             return metric

#         except optuna.TrialPruned:
#             raise
#         except Exception as e:
#             print(f"[TRIAL FAILED] {cfg_dict['name']}: {type(e).__name__}: {e}")
#             raise optuna.TrialPruned(str(e))

#         finally:
#             if pipeline is not None:
#                 del pipeline
#             if config is not None:
#                 del config
#             gc.collect()
#             torch.cuda.empty_cache()

#     return objective


# def main():
#     print(f"══════════════════════════════════════════")
#     print(f"  ALGORITHMUS: {ALGORITHM}")
#     print(f"  DB:          {DB_PATH}")
#     print(f"  CSV:         {CSV_SUMMARY_PATH}")
#     print(f"  Best-Config: {BEST_CONFIG_PATH}")
#     print(f"══════════════════════════════════════════")

#     all_results = []

#     combos = list(itertools.product(GRID_HIDDEN_SIZE, GRID_NUM_HEADS, GRID_META_ITER))
#     total_runs = len(combos) * INNER_TRIALS_PER_COMBO
#     print(f"Gesamt: {len(combos)} Grid-Kombinationen × {INNER_TRIALS_PER_COMBO} TPE-Trials "
#           f"= {total_runs} Läufe insgesamt")

#     for hidden_size, num_heads, meta_iter in combos:
#         study_name = f"grid_{ALGO_TAG}_h{hidden_size}_nh{num_heads}_it{meta_iter}"
#         study = optuna.create_study(
#             direction="minimize",
#             study_name=study_name,
#             storage=DB_PATH,
#             load_if_exists=True,
#         )
#         objective = make_objective(hidden_size, num_heads, meta_iter)
#         study.optimize(
#             objective,
#             n_trials=INNER_TRIALS_PER_COMBO,
#             gc_after_trial=True,
#             catch=(RuntimeError,),
#         )

#         if len(study.trials) == 0 or study.best_trial is None:
#             print(f"[WARN] {study_name}: keine erfolgreichen Trials, überspringe.")
#             continue

#         all_results.append({
#             "algorithm": ALGORITHM,
#             "hidden_size": hidden_size,
#             "num_heads": num_heads,
#             "meta_training_iter": meta_iter,
#             "best_value": study.best_value,
#             "best_params": study.best_params,
#         })
#         print(f"[DONE] {study_name} → best_value={study.best_value:.6f}")

#     if not all_results:
#         raise RuntimeError(f"Keine erfolgreichen Trials für {ALGORITHM}. Abbruch.")

#     # ── Globales Beste über alle 45 Grid-Kombinationen ────────────
#     best_overall = min(all_results, key=lambda r: r["best_value"])
#     print("\n" + "=" * 60)
#     print(f"GLOBAL BEST ({ALGORITHM}):")
#     print(best_overall)
#     print("=" * 60)

#     # ── Bestes Config-YAML für finalen Full-Run exportieren ───────
#     best_cfg = load_base_config()
#     best_cfg["algorithm"] = ALGORITHM
#     best_cfg["model_config"]["config"]["hidden_size"] = best_overall["hidden_size"]
#     best_cfg["model_config"]["config"]["num_heads"] = best_overall["num_heads"]
#     best_cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE
#     best_cfg["meta_lr"] = best_overall["best_params"]["meta_lr"]
#     best_cfg["fast_lr"] = best_overall["best_params"]["fast_lr"]
#     best_cfg["weight_decay"] = best_overall["best_params"]["weight_decay"]
#     best_cfg["model_config"]["config"]["num_layers"] = best_overall["best_params"]["num_layers"]
#     best_cfg["model_config"]["config"]["dropout"] = best_overall["best_params"]["dropout"]
#     best_cfg["meta_training_iter"] = 200  # voller Lauf für finales Training
#     best_cfg["name"] = f"pv_timeseries_tft_v2_best_{ALGO_TAG}"

#     with open(BEST_CONFIG_PATH, "w") as f:
#         yaml.safe_dump(best_cfg, f)
#     print(f"→ {BEST_CONFIG_PATH} geschrieben.")

#     # ── Alle Ergebnisse als CSV für Analyse/Report ─────────────────
#     pd.DataFrame(all_results).to_csv(CSV_SUMMARY_PATH, index=False)
#     print(f"→ {CSV_SUMMARY_PATH} geschrieben.")


# if __name__ == "__main__":
#     main()

"""optuna_search.py — TFT"""
import gc
import itertools
import uuid
import yaml
import optuna
import torch
import pytorch_lightning as pl
import pandas as pd

from conf.configs import PipelineParams

BASE_CONFIG_PATH = "conf/yaml/pv_tft_v3.yaml"
METRIC_CANDIDATES = ["meta_val_loss_masked", "meta_val_loss_masked_epoch",
                      "meta_val_loss", "meta_val_loss_epoch"]

# ══════════════════════════════════════════════════════════════════
# NUR DIESE ZWEI ZEILEN ÄNDERN
# ══════════════════════════════════════════════════════════════════
ALGORITHM = "RL2"   # "ANIL" oder "MAML"
BASELINE = False        # True = Inner-Loop aktiv, False = kein Meta-Learn
# ══════════════════════════════════════════════════════════════════

ALGO_TAG = ALGORITHM.lower()
BASELINE_TAG = "baseline" if BASELINE else "metalearn"

# ── Vom Supervisor vorgegebenes, EXHAUSTIVES Grid ────────────────────
GRID_HIDDEN_SIZE = [8, 16, 32, 64, 128]
GRID_NUM_HEADS = [2, 4, 8]
GRID_META_ITER = [3, 5, 10]
FIXED_WINDOW_STRIDE = 12

# ── Anzahl TPE-Trials PRO Grid-Kombination ───────────────────────────
INNER_TRIALS_PER_COMBO = 8

# ── Algorithmus- und baseline-spezifische Output-Pfade ──────────────
DB_PATH = f"sqlite:///optuna_pv_tft_{ALGO_TAG}_{BASELINE_TAG}.db"
CSV_SUMMARY_PATH = f"optuna_grid_summary_tft_{ALGO_TAG}_{BASELINE_TAG}.csv"
BEST_CONFIG_PATH = f"conf/yaml/best_config_tft_{ALGO_TAG}_{BASELINE_TAG}.yaml"


def load_base_config() -> dict:
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_metric(callback_metrics: dict) -> float:
    for key in METRIC_CANDIDATES:
        if key in callback_metrics:
            val = callback_metrics[key]
            return float(val.item() if hasattr(val, "item") else val)
    raise KeyError(
        f"Keine der erwarteten Metriken gefunden. Verfügbare Keys: "
        f"{list(callback_metrics.keys())}"
    )


def build_trial_config(trial: optuna.Trial, hidden_size: int,
                        num_heads: int, meta_iter: int) -> dict:
    cfg = load_base_config()

    cfg["name"] = (f"pv_tft_optuna_{ALGO_TAG}_{BASELINE_TAG}_h{hidden_size}_nh{num_heads}_"
                   f"it{meta_iter}_{trial.number}_{uuid.uuid4().hex[:6]}")

    cfg["algorithm"] = ALGORITHM
    cfg["baseline"] = BASELINE

    # print_meta_val NICHT überschreiben -> bleibt True (aus base yaml)
    # -> state.train=False in validation_step -> korrekter, getesteter Code-Pfad

    cfg["model_config"]["config"]["hidden_size"] = hidden_size
    cfg["model_config"]["config"]["num_heads"] = num_heads
    cfg["meta_training_iter"] = meta_iter
    cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE

    cfg["meta_lr"] = trial.suggest_float("meta_lr", 1e-6, 5e-4, log=True)
    cfg["fast_lr"] = trial.suggest_float("fast_lr", 1e-3, 3e-1, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg["model_config"]["config"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    cfg["model_config"]["config"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)

    # adaptation_steps nur relevant, wenn Inner-Loop aktiv ist
    # if BASELINE:
    #     cfg["adaptation_steps"] = trial.suggest_int("adaptation_steps", 1, 10)

    cfg["validate_every"] = min(cfg.get("validate_every", 5), meta_iter)

    cfg["trainer_config"]["limit_train_batches"] = 200
    cfg["trainer_config"]["limit_val_batches"] = 50

    return cfg


def make_objective(hidden_size: int, num_heads: int, meta_iter: int):
    """Erzeugt eine objective()-Funktion für eine feste Grid-Kombination."""
    def objective(trial: optuna.Trial) -> float:
        cfg_dict = build_trial_config(trial, hidden_size, num_heads, meta_iter)

        tmp_config_path = f"/tmp/optuna_cfg_{cfg_dict['name']}.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        pipeline = None
        config = None
        try:
            config = PipelineParams(config_path=tmp_config_path, **cfg_dict)
            pl.seed_everything(cfg_dict.get("seed", 42), workers=True)

            pipeline = config.pipeline
            pipeline.meta_training()

            metric = get_metric(pipeline.trainer.callback_metrics)

            if metric != metric:  # NaN
                raise optuna.TrialPruned("NaN metric")

            return metric

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[TRIAL FAILED] {cfg_dict['name']}: {type(e).__name__}: {e}")
            raise optuna.TrialPruned(str(e))

        finally:
            if pipeline is not None:
                del pipeline
            if config is not None:
                del config
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def main():
    print(f"══════════════════════════════════════════")
    print(f"  ALGORITHMUS: {ALGORITHM}  ({BASELINE_TAG})")
    print(f"  DB:          {DB_PATH}")
    print(f"  CSV:         {CSV_SUMMARY_PATH}")
    print(f"  Best-Config: {BEST_CONFIG_PATH}")
    print(f"══════════════════════════════════════════")

    all_results = []

    combos = list(itertools.product(GRID_HIDDEN_SIZE, GRID_NUM_HEADS, GRID_META_ITER))
    total_runs = len(combos) * INNER_TRIALS_PER_COMBO
    print(f"Gesamt: {len(combos)} Grid-Kombinationen × {INNER_TRIALS_PER_COMBO} TPE-Trials "
          f"= {total_runs} Läufe insgesamt")

    for hidden_size, num_heads, meta_iter in combos:
        study_name = f"grid_{ALGO_TAG}_{BASELINE_TAG}_h{hidden_size}_nh{num_heads}_it{meta_iter}"
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=DB_PATH,
            load_if_exists=True,
        )
        objective = make_objective(hidden_size, num_heads, meta_iter)
        study.optimize(
            objective,
            n_trials=INNER_TRIALS_PER_COMBO,
            gc_after_trial=True,
            catch=(RuntimeError,),
        )

        if len(study.trials) == 0 or study.best_trial is None:
            print(f"[WARN] {study_name}: keine erfolgreichen Trials, überspringe.")
            continue

        all_results.append({
            "algorithm": ALGORITHM,
            "baseline": BASELINE_TAG,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "meta_training_iter": meta_iter,
            "best_value": study.best_value,
            "best_params": study.best_params,
        })
        print(f"[DONE] {study_name} → best_value={study.best_value:.6f}")

    if not all_results:
        raise RuntimeError(f"Keine erfolgreichen Trials für {ALGORITHM} ({BASELINE_TAG}). Abbruch.")

    # ── Globales Beste über alle Grid-Kombinationen ────────────
    best_overall = min(all_results, key=lambda r: r["best_value"])

    # ── Grenzwert-Warnungen: liegt das Optimum am Rand des Suchraums? ──
    if best_overall["hidden_size"] == max(GRID_HIDDEN_SIZE):
        print(f"[WARN] best hidden_size={best_overall['hidden_size']} liegt am oberen "
              f"Grid-Rand -> ggf. GRID_HIDDEN_SIZE weiter erhöhen und erneut suchen.")
    if best_overall["meta_training_iter"] == max(GRID_META_ITER):
        print(f"[WARN] best meta_training_iter={best_overall['meta_training_iter']} liegt am "
              f"oberen Grid-Rand -> ggf. GRID_META_ITER weiter erhöhen.")
    if best_overall["num_heads"] == max(GRID_NUM_HEADS):
        print(f"[WARN] best num_heads={best_overall['num_heads']} liegt am oberen "
              f"Grid-Rand -> ggf. GRID_NUM_HEADS weiter erhöhen.")
    if best_overall["best_params"]["fast_lr"] > 0.27:
        print(f"[WARN] best fast_lr={best_overall['best_params']['fast_lr']:.4f} nahe am "
              f"oberen Suchbereichsrand (0.3) -> ggf. Range erweitern.")
    if BASELINE and best_overall["best_params"].get("adaptation_steps", 0) >= 9:
        print(f"[WARN] best adaptation_steps={best_overall['best_params']['adaptation_steps']} "
              f"nahe am oberen Suchbereichsrand (10) -> ggf. Range erweitern.")

    print("\n" + "=" * 60)
    print(f"GLOBAL BEST ({ALGORITHM}, {BASELINE_TAG}):")
    print(best_overall)
    print("=" * 60)

    # ── Bestes Config-YAML für finalen Full-Run exportieren ───────
    best_cfg = load_base_config()
    best_cfg["algorithm"] = ALGORITHM
    best_cfg["baseline"] = BASELINE
    best_cfg["model_config"]["config"]["hidden_size"] = best_overall["hidden_size"]
    best_cfg["model_config"]["config"]["num_heads"] = best_overall["num_heads"]
    best_cfg["data_config"]["config"]["window_stride"] = FIXED_WINDOW_STRIDE
    best_cfg["meta_lr"] = best_overall["best_params"]["meta_lr"]
    best_cfg["fast_lr"] = best_overall["best_params"]["fast_lr"]
    best_cfg["weight_decay"] = best_overall["best_params"]["weight_decay"]
    best_cfg["model_config"]["config"]["num_layers"] = best_overall["best_params"]["num_layers"]
    best_cfg["model_config"]["config"]["dropout"] = best_overall["best_params"]["dropout"]
    if BASELINE and "adaptation_steps" in best_overall["best_params"]:
        best_cfg["adaptation_steps"] = best_overall["best_params"]["adaptation_steps"]
    best_cfg["meta_training_iter"] = 200  # voller Lauf für finales Training
    best_cfg["name"] = f"pv_timeseries_tft_v2_best_{ALGO_TAG}_{BASELINE_TAG}"

    with open(BEST_CONFIG_PATH, "w") as f:
        yaml.safe_dump(best_cfg, f)
    print(f"→ {BEST_CONFIG_PATH} geschrieben.")

    # ── Alle Ergebnisse als CSV für Analyse/Report ─────────────────
    pd.DataFrame(all_results).to_csv(CSV_SUMMARY_PATH, index=False)
    print(f"→ {CSV_SUMMARY_PATH} geschrieben.")


if __name__ == "__main__":
    main()