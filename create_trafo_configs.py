# # # # # create_and_fill_trafo_configs.py
# # # # import re
# # # # import yaml
# # # # from pathlib import Path

# # # # CONF_DIR = Path("conf/yaml")
# # # # OUTPUT_DIR = Path("output")
# # # # SRC_PATTERN = "best_config_*.yaml"
# # # # TRAFO_PATTERN = "best_config_trafo_*.yaml"

# # # # EXTRA_KNOWN_REALS = [
# # # #     "WIND_DEGREES_100M_SIN",
# # # #     "WIND_DEGREES_100M_COS",
# # # #     "WIND_SPEED_100M",
# # # # ]

# # # # CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?\.ckpt")

# # # # # Explicit mapping: algorithm -> seed66 run directory
# # # # ALGO_TO_RUNDIR = {
# # # #     "anil": "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
# # # #     "maml": "pv_timeseries_tft_v2_best_maml_seed66",
# # # #     "rl2":  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# # # # }


# # # # def activate_trafo(cfg: dict) -> dict:
# # # #     dcfg = cfg["data_config"]["config"]

# # # #     dcfg["trafo_dir"] = "data/trafo_data"
# # # #     dcfg["trafo_test_only"] = True
# # # #     dcfg["use_trafo_features"] = True
# # # #     dcfg["plants_dir"] = "data/PV_data_trafo_random"

# # # #     reals = dcfg.get("time_varying_known_reals", [])
# # # #     added = sum(1 for feat in EXTRA_KNOWN_REALS if feat not in reals)
# # # #     for feat in EXTRA_KNOWN_REALS:
# # # #         if feat not in reals:
# # # #             reals.append(feat)
# # # #     dcfg["time_varying_known_reals"] = reals

# # # #     mcfg = cfg["model_config"]["config"]
# # # #     mcfg["enc_size"] = mcfg.get("enc_size", 0) + added

# # # #     return cfg


# # # # def best_checkpoint(ckpt_dir: Path) -> Path | None:
# # # #     candidates = []
# # # #     for f in ckpt_dir.glob("*.ckpt"):
# # # #         m = CKPT_RE.match(f.name)
# # # #         if not m:
# # # #             continue
# # # #         epoch, step, version = m.groups()
# # # #         version = int(version) if version else -1
# # # #         candidates.append((int(epoch), int(step), version, f))

# # # #     if not candidates:
# # # #         return None

# # # #     candidates.sort(key=lambda x: (x[0], x[1], x[2]))
# # # #     return candidates[-1][3]


# # # # def detect_algo(variant: str) -> str | None:
# # # #     for algo in ("anil", "maml", "rl2"):
# # # #         if algo in variant:
# # # #             return algo
# # # #     return None


# # # # def create_trafo_configs():
# # # #     files = sorted(CONF_DIR.glob(SRC_PATTERN))
# # # #     if not files:
# # # #         print(f"No files found matching {CONF_DIR / SRC_PATTERN}")
# # # #         return

# # # #     for f in files:
# # # #         if "trafo" in f.name:
# # # #             continue
# # # #         rest = f.name[len("best_config_"):]
# # # #         new_path = f.parent / f"best_config_trafo_{rest}"

# # # #         with f.open("r") as fh:
# # # #             cfg = yaml.safe_load(fh)

# # # #         cfg = activate_trafo(cfg)

# # # #         with new_path.open("w") as fh:
# # # #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# # # #         print(f"✓ Created {f.name} -> {new_path.name}")


# # # # # def fill_trafo_checkpoints():
# # # # #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# # # # #         variant = cfg_path.stem[len("best_config_trafo_"):]

# # # # #         # Only in scope: tft_* and lstm_* variants
# # # # #         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
# # # # #             continue

# # # # #         if "baseline" in variant:
# # # # #             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
# # # # #             continue

# # # # #         algo = detect_algo(variant)
# # # # #         if algo is None:
# # # # #             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm (anil/maml/rl2)")
# # # # #             continue

# # # # #         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
# # # # #         ckpt_dir = run_dir / "checkpoints"

# # # # #         if not ckpt_dir.is_dir():
# # # # #             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
# # # # #             continue

# # # # #         ckpt = best_checkpoint(ckpt_dir)
# # # # #         if ckpt is None:
# # # # #             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
# # # # #             continue

# # # # #         with cfg_path.open("r") as fh:
# # # # #             cfg = yaml.safe_load(fh)

# # # # #         cfg["checkpoint_to_load"] = str(ckpt)

# # # # #         if not cfg["name"].endswith("_trafo"):
# # # # #             cfg["name"] = f"{cfg['name']}_trafo"

# # # # #         with cfg_path.open("w") as fh:
# # # # #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# # # # #         print(f"✓ Updated {cfg_path.name}: algo={algo}, checkpoint={ckpt.name}, name={cfg['name']}")

# # # # def fill_trafo_checkpoints():
# # # #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# # # #         variant = cfg_path.stem[len("best_config_trafo_"):]

# # # #         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
# # # #             continue
# # # #         if "baseline" in variant:
# # # #             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
# # # #             continue

# # # #         algo = detect_algo(variant)
# # # #         if algo is None:
# # # #             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
# # # #             continue

# # # #         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
# # # #         ckpt_dir = run_dir / "checkpoints"

# # # #         if not ckpt_dir.is_dir():
# # # #             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
# # # #             continue

# # # #         ckpt = best_checkpoint(ckpt_dir)
# # # #         if ckpt is None:
# # # #             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
# # # #             continue

# # # #         with cfg_path.open("r") as fh:
# # # #             cfg = yaml.safe_load(fh)

# # # #         # --- correct split: directory vs. filename ---
# # # #         cfg["load_dir"] = str(ckpt_dir)
# # # #         cfg["checkpoint_to_load"] = ckpt.name

# # # #         # --- enable test-only mode ---
# # # #         cfg["meta_training_iter"] = 0

# # # #         if not cfg["name"].endswith("_trafo"):
# # # #             cfg["name"] = f"{cfg['name']}_trafo"

# # # #         with cfg_path.open("w") as fh:
# # # #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# # # #         print(f"✓ Updated {cfg_path.name}: algo={algo}, load_dir={ckpt_dir}, "
# # # #               f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")


# # # # def main():
# # # #     create_trafo_configs()
# # # #     print()
# # # #     fill_trafo_checkpoints()


# # # # if __name__ == "__main__":
# # # #     main()

# # # # create_and_fill_trafo_configs.py
# # # import re
# # # import yaml
# # # from pathlib import Path

# # # CONF_DIR = Path("conf/yaml")
# # # OUTPUT_DIR = Path("output")
# # # SRC_PATTERN = "best_config_*.yaml"
# # # TRAFO_PATTERN = "best_config_trafo_*.yaml"

# # # # Canonical full feature list, taken from conf/yaml/trafo_pv_tft.yaml
# # # TARGET_KNOWN_REALS = [
# # #     "WIND_DEGREES_40M_SIN",
# # #     "WIND_DEGREES_40M_COS",
# # #     "PRESSURE_SURFACE",
# # #     "DIRECT_HORIZONTAL_IRRADIANCE",
# # #     "WIND_SPEED_40M",
# # #     "TEMPERATURE_2METER_ABOVE_GROUND",
# # #     "HUMIDITY_RELATIVE",
# # #     "DIFFUSE_HORIZONTAL_IRRADIANCE",
# # #     "SOLAR_ELEVATION",
# # #     "SOLAR_AZIMUTH_SIN",
# # #     "SOLAR_AZIMUTH_COS",
# # #     "TIME_OF_DAY_SIN",
# # #     "TIME_OF_DAY_COS",
# # #     "DAY_OF_YEAR_SIN",
# # #     "DAY_OF_YEAR_COS",
# # #     "WIND_DEGREES_100M_SIN",
# # #     "WIND_DEGREES_100M_COS",
# # #     "WIND_SPEED_100M",
# # # ]

# # # CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?\.ckpt")

# # # ALGO_TO_RUNDIR = {
# # #     "anil": "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
# # #     "maml": "pv_timeseries_tft_v2_best_maml_seed66",
# # #     "rl2":  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# # # }


# # # def activate_trafo(cfg: dict, filename: str) -> dict:
# # #     dcfg = cfg["data_config"]["config"]

# # #     dcfg["trafo_dir"] = "data/trafo_data"
# # #     dcfg["trafo_test_only"] = True
# # #     dcfg["use_trafo_features"] = True
# # #     dcfg["plants_dir"] = "data/PV_data_trafo_random"

# # #     existing = dcfg.get("time_varying_known_reals", [])
# # #     missing  = [f for f in TARGET_KNOWN_REALS if f not in existing]
# # #     unknown  = [f for f in existing if f not in TARGET_KNOWN_REALS]

# # #     # Report what's happening per file, so nothing silently changes
# # #     print(f"  [{filename}] had {len(existing)}/{len(TARGET_KNOWN_REALS)} known reals")
# # #     if missing:
# # #         print(f"    + adding missing: {missing}")
# # #     if unknown:
# # #         print(f"    ⚠ unexpected/custom features found (kept untouched): {unknown}")

# # #     # Enforce the full canonical list (order consistent with reference config)
# # #     dcfg["time_varying_known_reals"] = TARGET_KNOWN_REALS.copy() + unknown

# # #     mcfg = cfg["model_config"]["config"]
# # #     old_enc = mcfg.get("enc_size", None)
# # #     mcfg["enc_size"] = len(dcfg["time_varying_known_reals"])
# # #     if old_enc != mcfg["enc_size"]:
# # #         print(f"    enc_size: {old_enc} -> {mcfg['enc_size']}")

# # #     return cfg


# # # def enforce_trafo_only_test_split(cfg: dict, filename: str) -> dict:
# # #     scfg = cfg.get("task_sampler_config", {}).get("config", {})
# # #     splits = scfg.get("splits")

# # #     if not splits or len(splits) != 3:
# # #         print(f"  ⚠ [{filename}] no valid 'splits' found — skipped")
# # #         return cfg

# # #     old_splits = list(splits)
# # #     splits[2] = 0.0
# # #     scfg["splits"] = splits

# # #     if old_splits != splits:
# # #         print(f"  [{filename}] splits: {old_splits} -> {splits}")

# # #     return cfg


# # # def best_checkpoint(ckpt_dir: Path) -> Path | None:
# # #     candidates = []
# # #     for f in ckpt_dir.glob("*.ckpt"):
# # #         m = CKPT_RE.match(f.name)
# # #         if not m:
# # #             continue
# # #         epoch, step, version = m.groups()
# # #         version = int(version) if version else -1
# # #         candidates.append((int(epoch), int(step), version, f))

# # #     if not candidates:
# # #         return None

# # #     candidates.sort(key=lambda x: (x[0], x[1], x[2]))
# # #     return candidates[-1][3]


# # # def detect_algo(variant: str) -> str | None:
# # #     for algo in ("anil", "maml", "rl2"):
# # #         if algo in variant:
# # #             return algo
# # #     return None


# # # def create_trafo_configs():
# # #     print("=" * 70)
# # #     print("STEP 1: Creating trafo configs (features + splits)")
# # #     print("=" * 70)

# # #     files = sorted(CONF_DIR.glob(SRC_PATTERN))
# # #     if not files:
# # #         print(f"No files found matching {CONF_DIR / SRC_PATTERN}")
# # #         return

# # #     for f in files:
# # #         if "trafo" in f.name:
# # #             continue
# # #         rest = f.name[len("best_config_"):]
# # #         new_path = f.parent / f"best_config_trafo_{rest}"

# # #         with f.open("r") as fh:
# # #             cfg = yaml.safe_load(fh)

# # #         cfg = activate_trafo(cfg, f.name)
# # #         cfg = enforce_trafo_only_test_split(cfg, f.name)

# # #         with new_path.open("w") as fh:
# # #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# # #         print(f"✓ Created {f.name} -> {new_path.name}\n")


# # # def fill_trafo_checkpoints():
# # #     print("=" * 70)
# # #     print("STEP 2: Filling checkpoints (tft_*/lstm_* only)")
# # #     print("=" * 70)

# # #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# # #         variant = cfg_path.stem[len("best_config_trafo_"):]

# # #         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
# # #             continue
# # #         if "baseline" in variant:
# # #             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
# # #             continue

# # #         algo = detect_algo(variant)
# # #         if algo is None:
# # #             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
# # #             continue

# # #         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
# # #         ckpt_dir = run_dir / "checkpoints"

# # #         if not ckpt_dir.is_dir():
# # #             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
# # #             continue

# # #         ckpt = best_checkpoint(ckpt_dir)
# # #         if ckpt is None:
# # #             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
# # #             continue

# # #         with cfg_path.open("r") as fh:
# # #             cfg = yaml.safe_load(fh)

# # #         cfg["load_dir"] = str(ckpt_dir)
# # #         cfg["checkpoint_to_load"] = ckpt.name
# # #         cfg["meta_training_iter"] = 0

# # #         if not cfg["name"].endswith("_trafo"):
# # #             cfg["name"] = f"{cfg['name']}_trafo"

# # #         with cfg_path.open("w") as fh:
# # #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# # #         print(f"✓ Updated {cfg_path.name}: algo={algo}, load_dir={ckpt_dir}, "
# # #               f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")


# # # def verify_known_reals():
# # #     print("\n" + "=" * 70)
# # #     print("STEP 3: Verification — time_varying_known_reals per trafo config")
# # #     print("=" * 70)

# # #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# # #         with cfg_path.open("r") as fh:
# # #             cfg = yaml.safe_load(fh)

# # #         reals = cfg["data_config"]["config"].get("time_varying_known_reals", [])
# # #         enc_size = cfg["model_config"]["config"].get("enc_size")
# # #         missing = [f for f in TARGET_KNOWN_REALS if f not in reals]

# # #         status = "✓ complete" if not missing else f"✗ MISSING {missing}"
# # #         print(f"  {cfg_path.name:55s} reals={len(reals):2d} enc_size={enc_size:2d}  {status}")


# # # def main():
# # #     create_trafo_configs()
# # #     print()
# # #     fill_trafo_checkpoints()
# # #     verify_known_reals()


# # # if __name__ == "__main__":
# # #     main()

# # # create_and_fill_trafo_configs.py
# # import re
# # import yaml
# # from pathlib import Path

# # CONF_DIR = Path("conf/yaml")
# # OUTPUT_DIR = Path("output")
# # SRC_PATTERN = "best_config_*.yaml"
# # TRAFO_PATTERN = "best_config_trafo_*.yaml"

# # CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?\.ckpt")

# # ALGO_TO_RUNDIR = {
# #     "anil": "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
# #     "maml": "pv_timeseries_tft_v2_best_maml_seed66",
# #     "rl2":  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# # }


# # def activate_trafo(cfg: dict, filename: str) -> dict:
# #     """Minimal-invasive trafo activation:
# #     - Only redirect the TEST split to trafo_dir.
# #     - Do NOT touch plants_dir, known_reals, or enc_size.
# #     - use_trafo_features stays False -> same FEATURE_COLS as original
# #       training run, so checkpoint shapes remain compatible.
# #     """
# #     dcfg = cfg["data_config"]["config"]

# #     dcfg["trafo_dir"] = "data/trafo_data"
# #     dcfg["trafo_test_only"] = True
# #     dcfg["use_trafo_features"] = False  # explicit: no extra columns

# #     existing = dcfg.get("time_varying_known_reals", [])
# #     mcfg = cfg["model_config"]["config"]
# #     enc_size = mcfg.get("enc_size")

# #     print(f"  [{filename}] known_reals={len(existing)} (unchanged), "
# #           f"enc_size={enc_size} (unchanged), use_trafo_features=False")

# #     return cfg


# # def enforce_trafo_only_test_split(cfg: dict, filename: str) -> dict:
# #     scfg = cfg.get("task_sampler_config", {}).get("config", {})
# #     splits = scfg.get("splits")

# #     if not splits or len(splits) != 3:
# #         print(f"  ⚠ [{filename}] no valid 'splits' found — skipped")
# #         return cfg

# #     old_splits = list(splits)
# #     splits[2] = 0.0
# #     scfg["splits"] = splits

# #     if old_splits != splits:
# #         print(f"  [{filename}] splits: {old_splits} -> {splits}")

# #     return cfg


# # def best_checkpoint(ckpt_dir: Path) -> Path | None:
# #     candidates = []
# #     for f in ckpt_dir.glob("*.ckpt"):
# #         m = CKPT_RE.match(f.name)
# #         if not m:
# #             continue
# #         epoch, step, version = m.groups()
# #         version = int(version) if version else -1
# #         candidates.append((int(epoch), int(step), version, f))

# #     if not candidates:
# #         return None

# #     candidates.sort(key=lambda x: (x[0], x[1], x[2]))
# #     return candidates[-1][3]


# # def detect_algo(variant: str) -> str | None:
# #     for algo in ("anil", "maml", "rl2"):
# #         if algo in variant:
# #             return algo
# #     return None


# # def create_trafo_configs():
# #     print("=" * 70)
# #     print("STEP 1: Creating trafo configs (test-split redirect only)")
# #     print("=" * 70)

# #     files = sorted(CONF_DIR.glob(SRC_PATTERN))
# #     if not files:
# #         print(f"No files found matching {CONF_DIR / SRC_PATTERN}")
# #         return

# #     for f in files:
# #         if "trafo" in f.name:
# #             continue
# #         rest = f.name[len("best_config_"):]
# #         new_path = f.parent / f"best_config_trafo_{rest}"

# #         with f.open("r") as fh:
# #             cfg = yaml.safe_load(fh)

# #         cfg = activate_trafo(cfg, f.name)
# #         cfg = enforce_trafo_only_test_split(cfg, f.name)

# #         with new_path.open("w") as fh:
# #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# #         print(f"✓ Created {f.name} -> {new_path.name}\n")


# # def fill_trafo_checkpoints():
# #     print("=" * 70)
# #     print("STEP 2: Filling checkpoints (tft_*/lstm_* only)")
# #     print("=" * 70)

# #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# #         variant = cfg_path.stem[len("best_config_trafo_"):]

# #         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
# #             continue
# #         if "baseline" in variant:
# #             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
# #             continue

# #         algo = detect_algo(variant)
# #         if algo is None:
# #             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
# #             continue

# #         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
# #         ckpt_dir = run_dir / "checkpoints"

# #         if not ckpt_dir.is_dir():
# #             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
# #             continue

# #         ckpt = best_checkpoint(ckpt_dir)
# #         if ckpt is None:
# #             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
# #             continue

# #         with cfg_path.open("r") as fh:
# #             cfg = yaml.safe_load(fh)

# #         cfg["load_dir"] = str(ckpt_dir)
# #         cfg["checkpoint_to_load"] = ckpt.name
# #         cfg["meta_training_iter"] = 0

# #         if not cfg["name"].endswith("_trafo"):
# #             cfg["name"] = f"{cfg['name']}_trafo"

# #         with cfg_path.open("w") as fh:
# #             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

# #         print(f"✓ Updated {cfg_path.name}: algo={algo}, load_dir={ckpt_dir}, "
# #               f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")


# # def verify_configs():
# #     print("\n" + "=" * 70)
# #     print("STEP 3: Verification — shape-relevant fields per trafo config")
# #     print("=" * 70)

# #     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
# #         with cfg_path.open("r") as fh:
# #             cfg = yaml.safe_load(fh)

# #         dcfg = cfg["data_config"]["config"]
# #         mcfg = cfg["model_config"]["config"]

# #         reals = dcfg.get("time_varying_known_reals", [])
# #         enc_size = mcfg.get("enc_size")
# #         use_trafo_feat = dcfg.get("use_trafo_features")
# #         trafo_test_only = dcfg.get("trafo_test_only")
# #         splits = cfg.get("task_sampler_config", {}).get("config", {}).get("splits")

# #         print(f"  {cfg_path.name:50s} "
# #               f"reals={len(reals):2d} enc_size={enc_size:>4} "
# #               f"use_trafo_feat={use_trafo_feat!s:5s} "
# #               f"trafo_test_only={trafo_test_only!s:5s} "
# #               f"splits={splits}")


# # def main():
# #     create_trafo_configs()
# #     print()
# #     fill_trafo_checkpoints()
# #     verify_configs()


# # if __name__ == "__main__":
# #     main()


# # create_and_fill_trafo_configs.py
# import re
# import yaml
# from pathlib import Path

# CONF_DIR = Path("conf/yaml")
# OUTPUT_DIR = Path("output")
# SRC_PATTERN = "best_config_*.yaml"
# TRAFO_PATTERN = "best_config_trafo_*.yaml"

# CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?\.ckpt")

# ALGO_TO_RUNDIR = {
#     "anil": "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
#     "maml": "pv_timeseries_tft_v2_best_maml_seed66",
#     "rl2":  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# }

# # Different model classes name their "number of input features" field differently
# SIZE_FIELD_CANDIDATES = ["enc_size", "input_size"]


# def get_size_field(mcfg: dict) -> tuple[str | None, int | None]:
#     """Return (field_name, value) for whichever size field exists in model_config."""
#     for field in SIZE_FIELD_CANDIDATES:
#         if field in mcfg:
#             return field, mcfg[field]
#     return None, None


# def activate_trafo(cfg: dict, filename: str) -> dict:
#     """Minimal-invasive trafo activation:
#     - Only redirect the TEST split to trafo_dir.
#     - Do NOT touch plants_dir, known_reals, or enc_size/input_size.
#     - use_trafo_features stays False -> same FEATURE_COLS as original
#       training run, so checkpoint shapes remain compatible.
#     """
#     dcfg = cfg["data_config"]["config"]

#     dcfg["trafo_dir"] = "data/trafo_data"
#     dcfg["trafo_test_only"] = True
#     dcfg["use_trafo_features"] = False  # explicit: no extra columns

#     existing = dcfg.get("time_varying_known_reals", [])
#     mcfg = cfg["model_config"]["config"]
#     size_field, size_val = get_size_field(mcfg)

#     print(f"  [{filename}] known_reals={len(existing)} (unchanged), "
#           f"{size_field or '???'}={size_val} (unchanged), use_trafo_features=False")

#     return cfg

# def enforce_trafo_only_test_split(cfg: dict, filename: str) -> dict:
#     scfg = cfg.get("task_sampler_config", {}).get("config", {})
#     splits = scfg.get("splits")

#     if not splits or len(splits) != 3:
#         print(f"  ⚠ [{filename}] no valid 'splits' found — skipped")
#         return cfg

#     old_splits = list(splits)

#     # test fraction -> 0.0 (trafo-only), val absorbs the leftover
#     # so the sum stays exactly 1.0
#     splits[1] = round(1.0 - splits[0], 10)
#     splits[2] = 0.0
#     scfg["splits"] = splits

#     if old_splits != splits:
#         print(f"  [{filename}] splits: {old_splits} -> {splits}")

#     return cfg


# def best_checkpoint(ckpt_dir: Path) -> Path | None:
#     candidates = []
#     for f in ckpt_dir.glob("*.ckpt"):
#         m = CKPT_RE.match(f.name)
#         if not m:
#             continue
#         epoch, step, version = m.groups()
#         version = int(version) if version else -1
#         candidates.append((int(epoch), int(step), version, f))

#     if not candidates:
#         return None

#     candidates.sort(key=lambda x: (x[0], x[1], x[2]))
#     return candidates[-1][3]


# def detect_algo(variant: str) -> str | None:
#     for algo in ("anil", "maml", "rl2"):
#         if algo in variant:
#             return algo
#     return None


# def create_trafo_configs():
#     print("=" * 70)
#     print("STEP 1: Creating trafo configs (test-split redirect only)")
#     print("=" * 70)

#     files = sorted(CONF_DIR.glob(SRC_PATTERN))
#     if not files:
#         print(f"No files found matching {CONF_DIR / SRC_PATTERN}")
#         return

#     for f in files:
#         if "trafo" in f.name:
#             continue
#         rest = f.name[len("best_config_"):]
#         new_path = f.parent / f"best_config_trafo_{rest}"

#         with f.open("r") as fh:
#             cfg = yaml.safe_load(fh)

#         cfg = activate_trafo(cfg, f.name)
#         cfg = enforce_trafo_only_test_split(cfg, f.name)

#         with new_path.open("w") as fh:
#             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

#         print(f"✓ Created {f.name} -> {new_path.name}\n")


# def fill_trafo_checkpoints():
#     print("=" * 70)
#     print("STEP 2: Filling checkpoints (tft_*/lstm_* only)")
#     print("=" * 70)

#     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
#         variant = cfg_path.stem[len("best_config_trafo_"):]

#         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
#             continue
#         if "baseline" in variant:
#             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
#             continue

#         algo = detect_algo(variant)
#         if algo is None:
#             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
#             continue

#         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
#         ckpt_dir = run_dir / "checkpoints"

#         if not ckpt_dir.is_dir():
#             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
#             continue

#         ckpt = best_checkpoint(ckpt_dir)
#         if ckpt is None:
#             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
#             continue

#         with cfg_path.open("r") as fh:
#             cfg = yaml.safe_load(fh)

#         cfg["load_dir"] = str(ckpt_dir)
#         cfg["checkpoint_to_load"] = ckpt.name
#         cfg["meta_training_iter"] = 0

#         if not cfg["name"].endswith("_trafo"):
#             cfg["name"] = f"{cfg['name']}_trafo"

#         with cfg_path.open("w") as fh:
#             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

#         print(f"✓ Updated {cfg_path.name}: algo={algo}, load_dir={ckpt_dir}, "
#               f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")


# def verify_configs():
#     print("\n" + "=" * 70)
#     print("STEP 3: Verification — shape-relevant fields per trafo config")
#     print("=" * 70)

#     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
#         with cfg_path.open("r") as fh:
#             cfg = yaml.safe_load(fh)

#         dcfg = cfg["data_config"]["config"]
#         mcfg = cfg["model_config"]["config"]

#         reals = dcfg.get("time_varying_known_reals", [])
#         size_field, size_val = get_size_field(mcfg)
#         size_str = f"{size_field}={size_val}" if size_field else "size=???"

#         use_trafo_feat = dcfg.get("use_trafo_features")
#         trafo_test_only = dcfg.get("trafo_test_only")
#         splits = cfg.get("task_sampler_config", {}).get("config", {}).get("splits")

#         print(f"  {cfg_path.name:50s} "
#               f"reals={len(reals):2d} {size_str:>14s} "
#               f"use_trafo_feat={str(use_trafo_feat):5s} "
#               f"trafo_test_only={str(trafo_test_only):5s} "
#               f"splits={splits}")


# def main():
#     create_trafo_configs()
#     print()
#     fill_trafo_checkpoints()
#     verify_configs()


# if __name__ == "__main__":
#     main()

import re
import yaml
from pathlib import Path

CONF_DIR = Path("conf/yaml")
OUTPUT_DIR = Path("output")
SRC_PATTERN = "best_config_*.yaml"
TRAFO_PATTERN = "best_config_trafo_*.yaml"

CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?\.ckpt")

# ALGO_TO_RUNDIR = {
#     "anil": "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
#     "maml": "pv_timeseries_tft_v2_best_maml_seed66",
#     "rl2":  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# }
# ALGO_TO_RUNDIR = {
#     ("anil", "lstm"): "pv_timeseries_lstm_v2_best_anil_metalearn_seed66",
#     ("maml", "lstm"): "pv_timeseries_lstm_v2_best_maml_metalearn_seed66",   # ← korrigiert
#     ("rl2",  "lstm"): "pv_timeseries_lstm_v2_best_rl2_metalearn_seed66",
#     ("anil", "tft"):  "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
#     ("maml", "tft"):  "pv_timeseries_tft_v2_best_maml_seed66",
#     ("rl2",  "tft"):  "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",
# }

ALGO_TO_RUNDIR = {
    # metalearn / normal runs
    ("anil", "lstm", False): "pv_timeseries_lstm_v2_best_anil_metalearn_seed66",
    ("maml", "lstm", False): "pv_timeseries_lstm_v2_best_maml_metalearn_seed66",
    ("rl2",  "lstm", False): "pv_timeseries_lstm_v2_best_rl2_metalearn_seed66",
    ("anil", "tft",  False): "pv_timeseries_tft_v2_best_anil_metalearn_seed66",
    ("maml", "tft",  False): "pv_timeseries_tft_v2_best_maml_seed66",
    ("rl2",  "tft",  False): "pv_timeseries_tft_v2_best_rl2_metalearn_seed66",

    # baseline runs
    ("maml", "lstm", True): "pv_timeseries_lstm_v2_best_maml_baseline_seed66",
    ("maml", "tft",  True): "pv_timeseries_tft_v2_best_maml_baseline_seed66",
    # ("anil", "lstm", True): "...",  # ← add once/if an anil baseline_seed66 dir exists
    # ("anil", "tft",  True): "...",
    # ("rl2",  "lstm", True): "...",  # ← add once/if a rl2 baseline_seed66 dir exists
    # ("rl2",  "tft",  True): "...",
}

SIZE_FIELD_CANDIDATES = ["enc_size", "input_size"]
SIZE_FIELD_CANDIDATES = ["enc_size", "input_size"]

# ── Device control ────────────────────────────────────────────
# Set to "cpu" or "cuda". This is the ONLY field that matters —
# PipelineParams.create_trainer() derives trainer_config.accelerator
# from this top-level 'device' field and overwrites whatever is in
# trainer_config.accelerator anyway. So don't bother setting that
# directly in the YAML; control it here instead.
TARGET_DEVICE = "cpu"   # <-- change to "cuda" if you want GPU


def set_device(cfg: dict, filename: str, device: str = TARGET_DEVICE) -> dict:
    """Force the top-level 'device' field, since this is what
    PipelineParams.create_trainer() actually reads to set
    trainer_config['accelerator']. Setting trainer_config.accelerator
    directly in the YAML has NO effect — it gets overwritten.
    """
    old_device = cfg.get("device")
    cfg["device"] = device

    # Optional cosmetic consistency — purely informational, since
    # the real value is recomputed from `device` at runtime anyway.
    if "trainer_config" in cfg and isinstance(cfg["trainer_config"], dict):
        cfg["trainer_config"]["accelerator"] = "gpu" if device == "cuda" else "cpu"

    if old_device != device:
        print(f"  [{filename}] device: {old_device} -> {device}")

    return cfg
# ──────────────────────────────────────────────────────────────


def get_size_field(mcfg: dict) -> tuple[str | None, int | None]:
    for field in SIZE_FIELD_CANDIDATES:
        if field in mcfg:
            return field, mcfg[field]
    return None, None


def activate_trafo(cfg: dict, filename: str) -> dict:
    dcfg = cfg["data_config"]["config"]

    dcfg["trafo_dir"] = "data/trafo_data"
    dcfg["trafo_test_only"] = True
    dcfg["use_trafo_features"] = False

    existing = dcfg.get("time_varying_known_reals", [])
    mcfg = cfg["model_config"]["config"]
    size_field, size_val = get_size_field(mcfg)

    print(f"  [{filename}] known_reals={len(existing)} (unchanged), "
          f"{size_field or '???'}={size_val} (unchanged), use_trafo_features=False")

    return cfg


def enforce_trafo_only_test_split(cfg: dict, filename: str) -> dict:
    scfg = cfg.get("task_sampler_config", {}).get("config", {})
    splits = scfg.get("splits")

    if not splits or len(splits) != 3:
        print(f"  ⚠ [{filename}] no valid 'splits' found — skipped")
        return cfg

    old_splits = list(splits)
    splits[1] = round(1.0 - splits[0], 10)
    splits[2] = 0.0
    scfg["splits"] = splits

    if old_splits != splits:
        print(f"  [{filename}] splits: {old_splits} -> {splits}")

    return cfg


def best_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = []
    for f in ckpt_dir.glob("*.ckpt"):
        m = CKPT_RE.match(f.name)
        if not m:
            continue
        epoch, step, version = m.groups()
        version = int(version) if version else -1
        candidates.append((int(epoch), int(step), version, f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[-1][3]


def detect_algo(variant: str) -> str | None:
    for algo in ("anil", "maml", "rl2"):
        if algo in variant:
            return algo
    return None


def create_trafo_configs():
    print("=" * 70)
    print("STEP 1: Creating trafo configs (test-split redirect + device)")
    print("=" * 70)

    files = sorted(CONF_DIR.glob(SRC_PATTERN))
    if not files:
        print(f"No files found matching {CONF_DIR / SRC_PATTERN}")
        return

    for f in files:
        if "trafo" in f.name:
            continue
        rest = f.name[len("best_config_"):]
        new_path = f.parent / f"best_config_trafo_{rest}"

        with f.open("r") as fh:
            cfg = yaml.safe_load(fh)

        cfg = activate_trafo(cfg, f.name)
        cfg = enforce_trafo_only_test_split(cfg, f.name)
        cfg = set_device(cfg, f.name)   # ← device control applied here

        with new_path.open("w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

        print(f"✓ Created {f.name} -> {new_path.name}\n")

def fill_trafo_checkpoints():
    print("=" * 70)
    print("STEP 2: Filling checkpoints (tft_*/lstm_* only, incl. baselines)")
    print("=" * 70)

    for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
        variant = cfg_path.stem[len("best_config_trafo_"):]

        if not (variant.startswith("tft_") or variant.startswith("lstm_")):
            continue

        is_baseline = "baseline" in variant
        model_type = "lstm" if variant.startswith("lstm_") else "tft"
        algo = detect_algo(variant)

        if algo is None:
            print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
            continue

        key = (algo, model_type, is_baseline)
        run_dir_name = ALGO_TO_RUNDIR.get(key)

        if run_dir_name is None:
            print(f"⚠ Skipped {cfg_path.name}: no run_dir entry for {key}")
            continue

        ckpt_dir = OUTPUT_DIR / run_dir_name / "checkpoints"
        if not ckpt_dir.is_dir():
            print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
            continue

        ckpt = best_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
            continue

        with cfg_path.open("r") as fh:
            cfg = yaml.safe_load(fh)

        cfg["load_dir"] = str(ckpt_dir)
        cfg["checkpoint_to_load"] = ckpt.name
        cfg["meta_training_iter"] = 0

        if not cfg["name"].endswith("_trafo"):
            cfg["name"] = f"{cfg['name']}_trafo"

        with cfg_path.open("w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

        print(f"✓ Updated {cfg_path.name}: algo={algo}, model_type={model_type}, "
              f"baseline={is_baseline}, load_dir={ckpt_dir}, "
              f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")

# def fill_trafo_checkpoints():
#     print("=" * 70)
#     print("STEP 2: Filling checkpoints (tft_*/lstm_* only)")
#     print("=" * 70)

#     for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
#         variant = cfg_path.stem[len("best_config_trafo_"):]

#         if not (variant.startswith("tft_") or variant.startswith("lstm_")):
#             continue
#         if "baseline" in variant:
#             print(f"⚠ Skipped {cfg_path.name}: baseline variant, no seed66 checkpoint available")
#             continue

#         algo = detect_algo(variant)
#         if algo is None:
#             print(f"⚠ Skipped {cfg_path.name}: could not detect algorithm")
#             continue
#         model_type = "lstm" if variant.startswith("lstm_") else "tft"
#         run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[(algo, model_type)]

#         # run_dir = OUTPUT_DIR / ALGO_TO_RUNDIR[algo]
#         ckpt_dir = run_dir / "checkpoints"

#         if not ckpt_dir.is_dir():
#             print(f"⚠ Skipped {cfg_path.name}: {ckpt_dir} not found")
#             continue

#         ckpt = best_checkpoint(ckpt_dir)
#         if ckpt is None:
#             print(f"⚠ Skipped {cfg_path.name}: no matching checkpoints in {ckpt_dir}")
#             continue

#         with cfg_path.open("r") as fh:
#             cfg = yaml.safe_load(fh)

#         cfg["load_dir"] = str(ckpt_dir)
#         cfg["checkpoint_to_load"] = ckpt.name
#         cfg["meta_training_iter"] = 0

#         if not cfg["name"].endswith("_trafo"):
#             cfg["name"] = f"{cfg['name']}_trafo"

#         with cfg_path.open("w") as fh:
#             yaml.safe_dump(cfg, fh, sort_keys=True, default_flow_style=False)

#         print(f"✓ Updated {cfg_path.name}: algo={algo}, load_dir={ckpt_dir}, "
#               f"checkpoint={ckpt.name}, meta_training_iter=0, name={cfg['name']}")


def verify_configs():
    print("\n" + "=" * 70)
    print("STEP 3: Verification — shape-relevant fields per trafo config")
    print("=" * 70)

    for cfg_path in sorted(CONF_DIR.glob(TRAFO_PATTERN)):
        with cfg_path.open("r") as fh:
            cfg = yaml.safe_load(fh)

        dcfg = cfg["data_config"]["config"]
        mcfg = cfg["model_config"]["config"]

        reals = dcfg.get("time_varying_known_reals", [])
        size_field, size_val = get_size_field(mcfg)
        size_str = f"{size_field}={size_val}" if size_field else "size=???"

        use_trafo_feat = dcfg.get("use_trafo_features")
        trafo_test_only = dcfg.get("trafo_test_only")
        splits = cfg.get("task_sampler_config", {}).get("config", {}).get("splits")
        device = cfg.get("device")
        accel = cfg.get("trainer_config", {}).get("accelerator")

        print(f"  {cfg_path.name:50s} "
              f"reals={len(reals):2d} {size_str:>14s} "
              f"use_trafo_feat={str(use_trafo_feat):5s} "
              f"trafo_test_only={str(trafo_test_only):5s} "
              f"device={device:4s} accelerator={accel} "
              f"splits={splits}")


def main():
    create_trafo_configs()
    print()
    fill_trafo_checkpoints()
    verify_configs()


if __name__ == "__main__":
    main()