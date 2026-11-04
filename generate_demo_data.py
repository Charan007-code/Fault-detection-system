"""
generate_demo_data.py - generated demo data for trying the software end-to-end.

    python generate_demo_data.py

!!! This data is invented by this script. A model trained on it proves that
!!! the code runs end-to-end. It proves NOTHING about detecting real network
!!! faults, and its accuracy must never be reported as a project result.

The rules below (e.g. "congestion = high latency + high throughput") are
hand-written, so a Random Forest will simply re-learn them. That is why the
accuracy on this demo data is high and not a real result.

It is written to dataset/demo/ and is excluded from real-data folder scans.
"""
import os

import numpy as np
import pandas as pd

SEED = 42
ROWS_PER_CLASS = 600
OUT = os.path.join("dataset", "demo", "demo_network_data.csv")

# Column names deliberately match what monitor.py can measure on a laptop,
# so the live "fill from monitor" demo works with the demo model.
PROFILES = {
    #               latency_ms   jitter_ms  loss_pct    throughput_mbps  if_errors/s  cpu%
    "Normal":       ((20, 8),    (2, 1),    (0.1, 0.2), (40, 15),        (0, 0.3),    (25, 10)),
    "Congestion":   ((120, 40),  (25, 10),  (2, 1.5),   (95, 10),        (0.5, 0.5),  (55, 15)),
    "High Latency": ((250, 60),  (8, 4),    (0.5, 0.5), (35, 15),        (0, 0.3),    (25, 10)),
    "Packet Loss":  ((40, 15),   (10, 5),   (15, 7),    (25, 12),        (3, 2),      (30, 10)),
    "Link Failure": ((900, 150), (0, 0.5),  (95, 5),    (0.5, 0.5),      (8, 4),      (20, 8)),
    "Node Failure": ((600, 200), (40, 20),  (60, 20),   (5, 4),          (2, 2),      (95, 4)),
}
COLUMNS = ["latency_ms", "jitter_ms", "packet_loss_pct", "throughput_mbps",
           "interface_errors_per_s", "cpu_percent"]
BOUNDS = {"packet_loss_pct": (0, 100), "cpu_percent": (0, 100)}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for label, params in PROFILES.items():
        block = {c: rng.normal(m, s, ROWS_PER_CLASS) for c, (m, s) in zip(COLUMNS, params)}
        df = pd.DataFrame(block)
        df["fault_label"] = label
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    for c in COLUMNS:
        lo, hi = BOUNDS.get(c, (0, None))
        df[c] = df[c].clip(lower=lo, upper=hi).round(3)
    # 3% missing values so the imputation code is exercised
    mask = rng.random(df[COLUMNS].shape) < 0.03
    df[COLUMNS] = df[COLUMNS].mask(mask)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} demo rows to {OUT}")
    print("Reminder: results on this file are NOT evidence of real fault detection.")


if __name__ == "__main__":
    main()
