"""
Shared utility for logging hyperparameter-sweep results to CSV so the
report's four hyperparameter tables (DQN, REINFORCE, PPO, A2C) can be
generated directly from data instead of copied by hand.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def log_run(algo: str, run_id: int, hyperparams: dict, metrics: dict):
    """Append one experiment run's results to logs/<algo>_runs.csv."""
    path = os.path.join(LOGS_DIR, f"{algo}_runs.csv")
    row = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        **hyperparams,
        **metrics,
    }
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[logger] Logged {algo} run {run_id} -> {path}")
    return path
