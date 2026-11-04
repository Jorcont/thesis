"""
parse_job_log.py
Parses output/logs/tft_trafo_<job_id>.out and appends a summary row to a CSV.

Usage:
    python utils/parse_job_log.py output/logs/tft_trafo_285290.out
    python utils/parse_job_log.py output/logs/tft_trafo_285290.out --out results/job_configs.csv
"""

import re
import csv
import sys
import os
from pathlib import Path
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_log(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def extract(pattern: str, text: str, default=None, group: int = 1):
    m = re.search(pattern, text)
    return m.group(group).strip() if m else default


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    text = read_log(log_path)

    row = {}

    # ── Header block ─────────────────────────────────────────────────────────
    row["job_id"]    = extract(r"Job ID:\s+(\d+)", text, "unknown")
    row["job_name"]  = extract(r"Job Name:\s+(\S+)", text)
    row["node"]      = extract(r"Node:\s+(\S+)", text)
    row["start"]     = extract(r"Start:\s+(.+)", text)
    row["log_file"]  = os.path.basename(log_path)

    # ── Meta-learning ─────────────────────────────────────────────────────────
    row["algorithm"]         = extract(r"algorithm: (\w+)", text)
    row["first_order"]       = extract(r"first_order: (\w+)", text)
    row["meta_lr"]           = extract(r"meta_lr: ([0-9e.\-]+)", text)
    row["fast_lr"]           = extract(r"fast_lr: ([0-9e.\-]+)", text)
    row["adaptation_steps"]  = extract(r"adaptation_steps: (\d+)", text)
    row["meta_training_iter"]= extract(r"meta_training_iter: (\d+)", text)
    row["weight_decay"]      = extract(r"weight_decay: ([0-9e.\-]+)", text)
    row["seed"]              = extract(r">>> Random seed from YAML\s*:\s*(\d+)", text)

    # ── Model ─────────────────────────────────────────────────────────────────
    row["model_name"]    = extract(r"model_config:.*?name: (\w+)", text)
    row["hidden_size"]   = extract(r"'hidden_size': (\d+)", text)
    row["num_heads"]     = extract(r"'num_heads': (\d+)", text)
    row["num_layers"]    = extract(r"'num_layers': (\d+)", text)
    row["dropout"]       = extract(r"'dropout': ([0-9.]+)", text)
    row["output_size"]   = extract(r"'output_size': (\d+)", text)   # n_quantiles
    row["enc_size"]      = extract(r"'enc_size': (\d+)", text)
    row["dec_size"]      = extract(r"'dec_size': (\d+)", text)

    # ── Loss ──────────────────────────────────────────────────────────────────
    row["loss_name"]     = extract(r"loss_config:.*?name: (\w+)", text)
    row["quantiles"]     = extract(r"'quantiles': (\[[^\]]+\])", text)
    row["median_only"]   = extract(r"'median_only': (\w+)", text)
    row["peak_alpha"]    = extract(r"'peak_alpha': ([0-9.]+)", text, default="2.5")

    # ── Data ──────────────────────────────────────────────────────────────────
    row["look_back"]         = extract(r"'look_back_steps': (\d+)", text)
    row["look_ahead"]        = extract(r"'look_ahead_steps': (\d+)", text)
    row["window_stride"]     = extract(r"'window_stride': (\d+)", text)
    row["gap"]               = extract(r"'gap': (\d+)", text)
    row["train_ratio"]       = extract(r"'train_ratio': ([0-9.]+)", text)
    row["max_windows_train"] = extract(r"'max_windows_train': (\d+)", text)
    row["max_windows_val"]   = extract(r"'max_windows_val': (\d+)", text)
    row["max_windows_test"]  = extract(r"'max_windows_test': (\d+)", text)
    row["theta_support"]     = extract(r"'theta_support': ([0-9.]+)", text)
    row["theta_query"]       = extract(r"'theta_query': ([0-9.]+)", text)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    row["plants_found"]    = extract(r"Found (\d+) plants", text)
    row["plants_after_nan"]= extract(r"NaN filter: \d+ → (\d+) plants", text)
    row["plants_train"]    = extract(r"train: (\d+)", text)
    row["plants_val"]      = extract(r"val: (\d+)", text)

    # ── Trainer ───────────────────────────────────────────────────────────────
    row["limit_train_batches"] = extract(r"limit_train_batches: (\d+)", text)
    row["limit_val_batches"]   = extract(r"limit_val_batches: (\d+)", text)
    row["max_epochs"]          = extract(r"max_epochs: (\d+)", text)
    row["N_train_tasks"]       = extract(r"'N_train': (\d+)", text)

    # ── Timestamp of parse ────────────────────────────────────────────────────
    row["parsed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return row


# ── CSV writer ────────────────────────────────────────────────────────────────

def append_to_csv(row: dict, out_path: str) -> None:
    path    = Path(out_path)
    exists  = path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Saved job {row['job_id']} → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_file",          help="Path to .out log file")
    parser.add_argument("--out", default="output/job_configs.csv",
                        help="Output CSV path (default: output/job_configs.csv)")
    args = parser.parse_args()

    row = parse_log(args.log_file)
    append_to_csv(row, args.out)

    # ── Quick preview ─────────────────────────────────────────────────────────
    print("\n── Extracted config ──────────────────────────────────────")
    for k, v in row.items():
        print(f"  {k:<25} {v}")