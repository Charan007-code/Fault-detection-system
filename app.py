"""
app.py - Flask web server + REST API.

    python app.py      ->  http://127.0.0.1:5000

Routes
  GET  /                     dashboard
  GET  /api/monitor          live measurements (?host=8.8.8.8&save=1)
  POST /api/predict          classify one set of measurements
  GET  /api/history          stored records (?type=predictions|monitoring&limit=50)
  GET  /api/model-info       which model is loaded, its features, classes and metrics
  GET  /api/stats            prediction counts per label
  GET  /reports/<file>       confusion matrix / feature-importance images
"""
import math
import os
import re
import threading
from datetime import datetime, timezone

import joblib
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import database
import monitor
from preprocess import records_to_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_MODEL = os.path.join(BASE_DIR, "model", "fault_model.joblib")
SYNTH_MODEL = os.path.join(BASE_DIR, "model", "synthetic", "fault_model.joblib")
MAX_MISSING_FRACTION = 0.5   # refuse to predict if more than half the features are absent

# Severity is a RULE-BASED label attached to the predicted class (not learned
# by the model). Classes that don't match any rule get severity = null.
SEVERITY_RULES = [
    (re.compile(r"^(normal|benign|healthy|no[_ ]?fault|ok)$", re.I), "none"),
    (re.compile(r"node|router|device|line[_ ]?card", re.I), "critical"),
    (re.compile(r"link", re.I), "critical"),
    (re.compile(r"rout|misconfig|bgp|ospf", re.I), "high"),
    (re.compile(r"loss", re.I), "high"),
    (re.compile(r"congest", re.I), "medium"),
    (re.compile(r"latency|delay|jitter", re.I), "medium"),
    (re.compile(r"transient", re.I), "low"),
]

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
database.init_db()


def load_model():
    path = os.environ.get("MODEL_PATH")
    for candidate in ([path] if path else [REAL_MODEL, SYNTH_MODEL]):
        if candidate and os.path.exists(candidate):
            bundle = joblib.load(candidate)
            bundle["path"] = os.path.relpath(candidate, BASE_DIR)
            return bundle
    return None


MODEL = load_model()


def severity_for(label):
    for pattern, level in SEVERITY_RULES:
        if pattern.search(str(label)):
            return level
    return None


def _model_or_503():
    if MODEL is None:
        abort(503, description="No trained model found. Run train_model.py first.")
    return MODEL


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(503)
def json_error(err):
    return jsonify({"error": err.description}), err.code


# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/reports/<path:filename>")
def reports(filename):
    return send_from_directory(os.path.join(BASE_DIR, "reports"), filename)


@app.get("/api/model-info")
def model_info():
    if MODEL is None:
        return jsonify({"loaded": False,
                        "message": "No model trained yet. See README: python train_model.py"})
    m = MODEL["metrics"]
    return jsonify({
        "loaded": True,
        "path": MODEL["path"],
        "data_source": MODEL["data_source"],
        "trained_at": MODEL["trained_at"],
        "label_column": MODEL["label_column"],
        "classes": MODEL["classes"],
        "severity": {c: severity_for(c) for c in MODEL["classes"]},
        "numeric_features": MODEL["numeric_features"],
        "categorical_features": MODEL["categorical_features"],
        "categorical_values": MODEL["categorical_values"],
        "numeric_ranges": MODEL["numeric_ranges"],
        "metrics": {k: m[k] for k in ("accuracy", "baseline_accuracy", "precision_macro",
                                      "recall_macro", "f1_macro", "f1_weighted",
                                      "split_strategy", "split_notes", "n_train", "n_test")},
        "per_class": m["per_class"],
        "top_features": m["top_features"][:10],
        "report_images": {
            "confusion_matrix": f"/{MODEL['report_dir']}/confusion_matrix.png",
            "feature_importance": f"/{MODEL['report_dir']}/feature_importance.png",
        },
    })


@app.get("/api/monitor")
def api_monitor():
    host = request.args.get("host", os.environ.get("MONITOR_HOST", "8.8.8.8"))
    if not monitor.valid_host(host):
        abort(400, description="invalid host")
    try:
        sample = monitor.collect(host)
    except Exception as exc:  # noqa: BLE001
        abort(503, description=f"monitoring failed: {exc}")
    if request.args.get("save", "1") != "0":
        database.insert_monitoring(sample)
    add_coverage(sample)
    return jsonify(sample)


def add_coverage(sample):
    """Tell the user which of the model's features this monitor sample provides."""
    if MODEL is None:
        return sample
    feats = MODEL["numeric_features"] + MODEL["categorical_features"]
    usable = [f for f in feats if sample.get(f) is not None]
    sample["model_feature_coverage"] = {
        "available": usable,
        "missing": [f for f in feats if f not in usable],
        "can_predict": len(usable) >= len(feats) * (1 - MAX_MISSING_FRACTION),
    }
    return sample


class InputError(ValueError):
    """Bad prediction input -> HTTP 400 in the API, a log line in auto-monitoring."""


def validate_features(payload, model):
    """Return (clean_dict, missing, ignored) or raise InputError with a clear message."""
    if not isinstance(payload, dict):
        raise InputError("Body must be a JSON object, e.g. {\"features\": {...}}")
    feats = payload.get("features", payload)
    if not isinstance(feats, dict) or not feats:
        raise InputError("'features' must be a non-empty JSON object")
    numeric, categorical = model["numeric_features"], model["categorical_features"]
    known = set(numeric) | set(categorical)
    clean, errors = {}, []
    for name, value in feats.items():
        if name not in known or value is None or value == "":
            continue
        if name in numeric:
            if isinstance(value, bool):
                errors.append(f"{name}: booleans are not numbers")
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name}: '{value}' is not a number")
                continue
            if not math.isfinite(num):
                errors.append(f"{name}: must be a finite number")
                continue
            clean[name] = num
        else:
            if not isinstance(value, (str, int, float)):
                errors.append(f"{name}: must be a string")
                continue
            clean[name] = str(value)
    if errors:
        raise InputError("; ".join(errors))
    missing = [f for f in numeric + categorical if f not in clean]
    ignored = [k for k in feats if k not in known]
    if len(missing) > len(known) * MAX_MISSING_FRACTION:
        raise InputError(
            f"Too many required features missing ({len(missing)} of {len(known)}). "
            f"Missing: {missing[:15]}{' ...' if len(missing) > 15 else ''}. "
            "The model would be guessing from imputed values, so no prediction was made.")
    return clean, missing, ignored


@app.post("/api/predict")
def api_predict():
    model = _model_or_503()
    try:
        return jsonify(run_prediction(request.get_json(silent=True), model, trigger="manual"))
    except InputError as exc:
        abort(400, description=str(exc))


def run_prediction(payload, model, trigger="manual"):
    """Validate -> preprocess -> predict -> store. Shared by the API and auto-monitor."""
    clean, missing, ignored = validate_features(payload, model)
    X = records_to_frame(clean, model["numeric_features"], model["categorical_features"])
    pipe = model["pipeline"]
    proba = pipe.predict_proba(X)[0]
    best = int(proba.argmax())
    label = str(pipe.classes_[best])
    confidence = round(float(proba[best]), 4)
    probabilities = {str(c): round(float(p), 4) for c, p in zip(pipe.classes_, proba)}
    severity = severity_for(label)

    warnings = []
    if model["data_source"] == "synthetic":
        warnings.append("Model trained on SYNTHETIC data - prediction is a software demo, not a real diagnosis.")
    if missing:
        warnings.append(f"{len(missing)} feature(s) missing and filled with training medians: {missing}")
    for f, v in clean.items():
        lo_hi = model["numeric_ranges"].get(f)
        if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
            warnings.append(f"{f}={v} is outside the training range {lo_hi}; prediction is extrapolation.")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record_id = database.insert_prediction(ts, label, confidence, severity,
                                           model["data_source"], clean, probabilities, trigger)
    return {
        "id": record_id,
        "trigger": trigger,
        "timestamp": ts,
        "prediction": label,
        "confidence": confidence,
        "confidence_note": "Fraction of Random Forest trees voting for this class (not a calibrated probability).",
        "severity": severity,
        "severity_note": "Rule-based mapping from the class name, not learned." if severity else
                         "No severity rule matches this class.",
        "probabilities": probabilities,
        "model_source": model["data_source"],
        "missing_features": missing,
        "ignored_fields": ignored,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Automatic background monitoring
# --------------------------------------------------------------------------- #
AUTO = {"enabled": False, "interval_s": None, "host": None, "runs": 0,
        "last_run": None, "last_sample": None, "last_prediction": None,
        "last_skip_reason": None, "last_error": None, "alerts": []}
ALERT_LEVELS = {"high", "critical"}


def auto_monitor_tick(host):
    """One automatic cycle: measure, store, and predict IF the model can use the data."""
    sample = add_coverage(monitor.collect(host))
    database.insert_monitoring(sample)
    AUTO.update(last_run=sample["timestamp"], last_sample=sample, last_error=None)
    AUTO["runs"] += 1
    if MODEL is None:
        AUTO["last_skip_reason"] = "no trained model"
        return
    try:
        result = run_prediction({"features": sample}, MODEL, trigger="auto")
    except InputError as exc:
        # expected when the model was trained on measurements this computer can't take
        AUTO["last_skip_reason"] = f"prediction skipped: {exc}"
        return
    AUTO.update(last_prediction=result, last_skip_reason=None)
    if result["severity"] in ALERT_LEVELS:
        alert = {"timestamp": result["timestamp"], "prediction": result["prediction"],
                 "severity": result["severity"], "confidence": result["confidence"],
                 "model_source": result["model_source"]}
        AUTO["alerts"] = ([alert] + AUTO["alerts"])[:20]
        app.logger.warning("ALERT %s (%s, %.0f%% votes, %s model)", result["prediction"],
                           result["severity"], result["confidence"] * 100, result["model_source"])


def start_auto_monitor(interval_s, host):
    """Run auto_monitor_tick every `interval_s` seconds in a daemon thread."""
    if not monitor.valid_host(host):
        raise ValueError(f"invalid host {host!r}")
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                auto_monitor_tick(host)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive, show the error
                AUTO["last_error"] = str(exc)
                app.logger.error("auto-monitor failed: %s", exc)
            stop.wait(interval_s)

    AUTO.update(enabled=True, interval_s=interval_s, host=host)
    threading.Thread(target=loop, name="auto-monitor", daemon=True).start()
    return stop


@app.get("/api/status")
def api_status():
    """Auto-monitor state + recent high/critical alerts."""
    out = dict(AUTO)
    out["model_loaded"] = MODEL is not None
    out["model_source"] = MODEL["data_source"] if MODEL else None
    return jsonify(out)


@app.get("/api/history")
def api_history():
    kind = request.args.get("type", "predictions")
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except ValueError:
        abort(400, description="limit must be an integer")
    try:
        rows = database.get_history(kind, limit)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify({"type": kind, "count": len(rows), "records": rows})


@app.get("/api/stats")
def api_stats():
    return jsonify({"prediction_counts": database.get_label_counts()})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Network fault detection dashboard")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--auto-monitor", type=int, metavar="SECONDS",
                    default=int(os.environ.get("AUTO_MONITOR_INTERVAL", 0)),
                    help="measure (and predict when possible) every N seconds; 0 = off")
    ap.add_argument("--host-target", default=os.environ.get("MONITOR_HOST", "8.8.8.8"),
                    help="host to ping for automatic monitoring")
    args = ap.parse_args()

    if MODEL is None:
        print("WARNING: no model found - train one first (see README).")
    else:
        print(f"Loaded model: {MODEL['path']}  (data source: {MODEL['data_source'].upper()})")
    if args.auto_monitor > 0:
        start_auto_monitor(max(5, args.auto_monitor), args.host_target)
        print(f"Auto-monitoring {args.host_target} every {max(5, args.auto_monitor)} s")
    # debug reloader would start the monitor thread twice, so it stays off
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
