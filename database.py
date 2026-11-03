"""database.py - SQLite storage for monitoring samples and predictions."""
import json
import os
import sqlite3
from contextlib import closing

DB_PATH = os.environ.get("NFD_DB_PATH", os.path.join("database", "network_faults.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    host TEXT,
    method TEXT,
    latency_ms REAL,
    jitter_ms REAL,
    packet_loss_pct REAL,
    probe_failure_pct REAL,
    throughput_mbps REAL,
    interface_errors_per_s REAL,
    interface_drops_per_s REAL,
    cpu_percent REAL,
    memory_percent REAL,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL,
    severity TEXT,
    model_source TEXT,
    input_json TEXT,
    probabilities_json TEXT,
    trigger TEXT DEFAULT 'manual'
);
"""
MON_COLS = ["timestamp", "host", "method", "latency_ms", "jitter_ms", "packet_loss_pct",
            "probe_failure_pct", "throughput_mbps", "interface_errors_per_s",
            "interface_drops_per_s", "cpu_percent", "memory_percent"]


def _connect(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path=None):
    with closing(_connect(path)) as conn:
        conn.executescript(SCHEMA)
        # upgrade databases created by the first version of this project
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
        if "trigger" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN trigger TEXT DEFAULT 'manual'")
        conn.commit()


def insert_monitoring(sample, path=None):
    values = [sample.get(c) for c in MON_COLS] + [json.dumps(sample)]
    with closing(_connect(path)) as conn:
        conn.execute(f"INSERT INTO monitoring ({', '.join(MON_COLS)}, raw_json) "
                     f"VALUES ({', '.join('?' * (len(MON_COLS) + 1))})", values)
        conn.commit()


def insert_prediction(ts, label, confidence, severity, source, inputs, probs,
                      trigger="manual", path=None):
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (timestamp, predicted_label, confidence, severity, "
            "model_source, input_json, probabilities_json, trigger) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, label, confidence, severity, source, json.dumps(inputs), json.dumps(probs), trigger))
        conn.commit()
        return cur.lastrowid


def get_history(kind="predictions", limit=50, path=None):
    table = {"predictions": "predictions", "monitoring": "monitoring"}.get(kind)
    if table is None:
        raise ValueError("type must be 'predictions' or 'monitoring'")
    with closing(_connect(path)) as conn:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("input_json", "probabilities_json", "raw_json"):
            if k in d and d[k]:
                d[k.replace("_json", "")] = json.loads(d.pop(k))
            else:
                d.pop(k, None)
        out.append(d)
    return out


def get_label_counts(path=None):
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT predicted_label, COUNT(*) AS n FROM predictions "
                            "GROUP BY predicted_label ORDER BY n DESC").fetchall()
    return {r["predicted_label"]: r["n"] for r in rows}
