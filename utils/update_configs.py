#!/usr/bin/env python3
"""Passt best_config_*.yaml Dateien an:
- limit_test_batches / limit_val_batches werden auskommentiert
- metric_keys wird gesetzt
- validate_every: 5 wird gesetzt
"""

import sys
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096
yaml.indent(mapping=2, sequence=2, offset=0)

CONF_DIR = Path("conf/yaml")
KEYS_TO_COMMENT = ("limit_test_batches", "limit_val_batches")
DESIRED_METRIC_KEYS = ["losses", "mae", "mse", "rmse"]


def comment_out_keys(trainer_config):
    """Entfernt aktive limit_*_batches-Keys und fügt sie als Kommentar wieder ein."""
    comment_lines = []
    for key in KEYS_TO_COMMENT:
        if key in trainer_config:
            value = trainer_config[key]
            comment_lines.append(f"{key}: {value}")
            del trainer_config[key]

    if comment_lines and len(trainer_config) > 0:
        first_key = next(iter(trainer_config))
        existing = trainer_config.ca.items.get(first_key)
        # Kommentar vor den ersten verbleibenden Key setzen
        trainer_config.yaml_set_comment_before_after_key(
            first_key, before="\n".join(comment_lines)
        )


def process_file(path: Path):
    data = yaml.load(path)

    trainer_config = data.get("trainer_config")
    if trainer_config is not None:
        comment_out_keys(trainer_config)

    metrics_config = data.get("metrics_config")
    if metrics_config is not None:
        metrics_config["metric_keys"] = DESIRED_METRIC_KEYS
        metrics_config["sample_efficiency"] = True

    data["validate_every"] = 5
    data["inner_batch_size"] = None
    data["seed"] = 25
    data["meta_training_iter"] = 70
    data["adaptation_steps"] = 5

    yaml.dump(data, path)

def main():
    args = sys.argv[1:]
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(CONF_DIR.glob("best_config_*.yaml"))

    if not files:
        print("Keine passenden Dateien gefunden.")
        return

    for f in files:
        print(f"Bearbeite: {f}")
        process_file(f)


if __name__ == "__main__":
    main()
