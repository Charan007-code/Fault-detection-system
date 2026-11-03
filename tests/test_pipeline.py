"""Run with:  python -m pytest -q

These tests use small made-up frames to check the CODE (splits, encoding,
API validation). They are not evidence about any real dataset.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocess import (coerce_types, load_config, records_to_frame,  # noqa: E402
                        select_features, split_data)


def make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=n, freq="min").astype(str),
        "router": rng.choice(["R1", "R2", "R3", "R4", "R5"], n),
        "rtt": rng.normal(20, 5, n),
        "loss": rng.random(n),
        "proto": rng.choice(["icmp", "snmp"], n),
        "row_id": np.arange(n).astype(str),
        "constant": 1,
        "label": rng.choice(["A", "B"], n),
    })


def cfg(**kw):
    c = load_config(None)
    c.update(label_column="label", **kw)
    return c


def test_feature_selection_excludes_reserved_ids_and_constants():
    df = make_frame()
    num, cat, excluded = select_features(df, cfg(group_column="router", time_column="ts"))
    assert set(num) == {"rtt", "loss"}
    assert cat == ["proto"]
    assert "row_id" in excluded and "constant" in excluded
    assert "label" not in num + cat and "router" not in num + cat


def test_group_split_keeps_groups_apart():
    df = make_frame()
    c = cfg(group_column="router")
    tr, te, strategy, _ = split_data(df, df["label"].to_numpy(), c)
    assert strategy == "group"
    assert not set(df.loc[tr, "router"]) & set(df.loc[te, "router"])


def test_chronological_split_tests_on_the_future():
    df = make_frame().sample(frac=1, random_state=1).reset_index(drop=True)
    tr, te, strategy, _ = split_data(df, df["label"].to_numpy(), cfg(time_column="ts"))
    assert strategy == "chronological"
    assert pd.to_datetime(df.loc[tr, "ts"]).max() <= pd.to_datetime(df.loc[te, "ts"]).min()


def test_inference_frame_matches_training_layout():
    X = records_to_frame({"loss": "0.5", "proto": "never-seen", "extra": 9}, ["rtt", "loss"], ["proto"])
    assert list(X.columns) == ["rtt", "loss", "proto"]
    assert np.isnan(X.loc[0, "rtt"]) and X.loc[0, "loss"] == 0.5


def test_coerce_handles_inf_and_text_numbers():
    df = pd.DataFrame({"a": ["1", "inf", None], "b": ["x", None, "y"]})
    X = coerce_types(df, ["a"], ["b"])
    assert X["a"].isna().sum() == 2


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("proj")
    csv = tmp / "data.csv"
    make_frame(600).to_csv(csv, index=False)
    conf = tmp / "conf.json"
    conf.write_text(json.dumps({"data_path": str(csv), "label_column": "label",
                                "group_column": "router", "drop_columns": ["row_id"]}))
    # train a throwaway model in a temp working dir so real outputs aren't touched
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "train_model.py"), "--config", str(conf)],
                   cwd=tmp, env=env, check=True, capture_output=True)
    os.environ["MODEL_PATH"] = str(tmp / "model" / "fault_model.joblib")
    os.environ["NFD_DB_PATH"] = str(tmp / "test.db")
    import importlib

    import database
    importlib.reload(database)
    import app as app_module
    importlib.reload(app_module)
    return app_module.app.test_client()


def test_predict_ok(client):
    r = client.post("/api/predict", json={"features": {"rtt": 20, "loss": 0.3, "proto": "icmp"}})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["prediction"] in ("A", "B")
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_bad_input(client):
    assert client.post("/api/predict", data="nope", content_type="text/plain").status_code == 400
    assert client.post("/api/predict", json={"features": {"rtt": "abc"}}).status_code == 400
    assert client.post("/api/predict", json={"features": {"rtt": True}}).status_code == 400
    # only 1 of 3 features -> too much imputation, refused
    assert client.post("/api/predict", json={"features": {"rtt": 5}}).status_code == 400


def test_history_and_model_info(client):
    client.post("/api/predict", json={"rtt": 20, "loss": 0.3, "proto": "snmp"})
    h = client.get("/api/history?type=predictions").get_json()
    assert h["count"] >= 1
    assert client.get("/api/history?type=bogus").status_code == 400
    info = client.get("/api/model-info").get_json()
    assert info["data_source"] == "real" and info["metrics"]["split_strategy"] == "group"


# ---------------- automation safety ----------------
def test_source_file_is_never_a_default_feature():
    df = make_frame()
    df["source_file"] = np.where(df["label"] == "A", "link_fail.csv", "normal.csv")
    num, cat, excluded = select_features(df, cfg())
    assert "source_file" not in num + cat and "source_file" in excluded


def test_continuous_measurements_are_not_flagged_as_ids():
    from preprocess import _is_row_counter
    rng = np.random.default_rng(0)
    assert not _is_row_counter(pd.Series(rng.normal(20, 5, 500)), 500)
    assert _is_row_counter(pd.Series(np.arange(500)), 500)


def test_zip_slip_is_blocked(tmp_path):
    import zipfile

    import automate
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../../escape.csv", "a\n1\n")
    with pytest.raises(RuntimeError):
        automate.safe_extract(str(evil), str(tmp_path / "out"))
    assert not (tmp_path / "escape.csv").exists()


def test_auto_monitor_tick_skips_when_features_unavailable(client):
    import app as app_module
    # the test model needs rtt/loss/proto - the local monitor has none of them
    app_module.auto_monitor_tick("127.0.0.1")
    assert app_module.AUTO["runs"] >= 1
    assert "prediction skipped" in (app_module.AUTO["last_skip_reason"] or "")
