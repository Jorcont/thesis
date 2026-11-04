#!/usr/bin/env python3
"""
Fügt zu 'name:' in allen best_config*.yaml-Dateien den Suffix '_seed<seed>' hinzu.
Der Dateiname selbst wird NICHT verändert - nur der Inhalt (Feld 'name').
"""

import argparse
import re
from pathlib import Path

from ruamel.yaml import YAML

SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


def update_yaml_name(path: Path, dry_run: bool = False) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # verhindert Zeilenumbrüche beim Schreiben

    with path.open("r") as f:
        data = yaml.load(f)

    if "name" not in data or "seed" not in data:
        print(f"[SKIP] {path.name}: 'name' oder 'seed' fehlt")
        return

    base_name = SEED_SUFFIX_RE.sub("", str(data["name"]))
    seed = data["seed"]
    new_name = f"{base_name}_seed{seed}"

    if data["name"] == new_name:
        print(f"[OK]   {path.name}: bereits korrekt ({new_name})")
        return

    print(f"[CHANGE] {path.name}: '{data['name']}' -> '{new_name}'")

    if not dry_run:
        data["name"] = new_name
        with path.open("w") as f:
            yaml.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Setzt 'name' in best_config*.yaml auf <name>_seed<seed>."
    )
    parser.add_argument(
        "directory", type=str, nargs="?", default="conf/yaml",
        help="Verzeichnis mit den YAML-Dateien (Standard: conf/yaml)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Zeigt nur an, was geändert würde, ohne Dateien zu schreiben."
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob("best_config*.yaml"))

    if not files:
        print(f"Keine 'best_config*.yaml'-Dateien in {directory} gefunden.")
        return

    for file_path in files:
        update_yaml_name(file_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
