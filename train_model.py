"""
train_model.py - STEP 2: train + evaluate the Random Forest.

    python train_model.py                 # REAL data, settings from dataset_config.json
    python train_model.py --synthetic     # SYNTHETIC test data -> model/synthetic/

Outputs (real data):            model/fault_model.joblib, reports/*
Outputs (synthetic, separate):  model/synthetic/fault_model.joblib, reports/synthetic/*
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline  # noqa: E402

from preprocess import (build_preprocessor, coerce_types, load_config,  # noqa: E402
                        load_dataset, select_features, split_data)

SYNTHETIC_CONFIG = {
    "data_path": "dataset/synthetic/SYNTHETIC_network_data.csv",
    "label_column": "fault_label",
    "split_strategy": "stratified",
}


def save_confusion_matrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(6, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm


def save_feature_importance(names, importances, path, title, top=20):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35)))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#2a7de1")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [{"feature": names[i], "importance": float(importances[i])} for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dataset_config.json")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on the SYNTHETIC test file instead of real data")
    args = ap.parse_args()

    if args.synthetic:
        cfg = load_config(None)
        cfg.update(SYNTHETIC_CONFIG)
        source, model_dir, report_dir = "synthetic", "model/synthetic", "reports/synthetic"
        if not os.path.exists(cfg["data_path"]):
            sys.exit("Synthetic file missing. Run:  python generate_synthetic.py")
    else:
        cfg = load_config(args.config)
        source, model_dir, report_dir = "real", "model", "reports"
        if not cfg.get("label_column"):
            sys.exit("label_column is not set in dataset_config.json.\n"
                     "Run  python inspect_dataset.py  first and choose it from the real columns.")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    tag = "SYNTHETIC DATA - NOT A REAL RESULT" if source == "synthetic" else "real dataset"

    # 1. Load -------------------------------------------------------------- #
    print(f"[1/8] Loading data ({tag}) from {cfg['data_path']}")
    df, paths = load_dataset(cfg["data_path"], cfg.get("sample_rows"), cfg["random_state"])
    label = cfg["label_column"]
    if label not in df.columns:
        sys.exit(f"label_column '{label}' not in data. Columns are: {list(df.columns)}")
    for key in ("group_column", "time_column"):
        if cfg.get(key) and cfg[key] not in df.columns:
            sys.exit(f"{key} '{cfg[key]}' not in data. Columns are: {list(df.columns)}")
    print(f"      {len(df):,} rows x {df.shape[1]} columns from {len(paths)} file(s)")

    # 2. Explore ------------------------------------------------------------ #
    before = len(df)
    df = df[df[label].notna()].reset_index(drop=True)
    if len(df) < before:
        print(f"      dropped {before - len(df)} rows with no label (cannot learn from them)")
    y = df[label].astype(str).str.strip().to_numpy()
    dist = pd.Series(y).value_counts()
    print("[2/8] Class distribution (THESE are the only faults the model can detect):")
    for k, v in dist.items():
        print(f"      {k:30s} {v:8,d}  ({v / len(y):6.2%})")
    if dist.size < 2:
        sys.exit("Need at least 2 classes to train a classifier.")

    # 3-5. Missing values, encoding, feature selection --------------------- #
    numeric, categorical, excluded = select_features(df, cfg)
    print(f"[3/8] Features: {len(numeric)} numeric, {len(categorical)} categorical")
    for c, why in excluded.items():
        print(f"      excluded '{c}': {why}")
    X = coerce_types(df, numeric, categorical)
    miss = X.isna().mean()
    if miss.any():
        print("      missing values will be imputed (median / '__missing__') "
              f"in {int((miss > 0).sum())} column(s), learned from TRAIN data only")

    # 6. Split --------------------------------------------------------------- #
    train_idx, test_idx, strategy, notes = split_data(df, y, cfg)
    print(f"[4/8] Split: {strategy}  train={len(train_idx):,}  test={len(test_idx):,}")
    for n_ in notes:
        print("      NOTE:", n_)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 7. Train -------------------------------------------------------------- #
    print("[5/8] Training Random Forest ...")
    model = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1,
            random_state=cfg["random_state"], min_samples_leaf=2)),
    ])
    model.fit(X_train, y_train)

    # 8-11. Evaluate -------------------------------------------------------- #
    print("[6/8] Evaluating on the held-out test set")
    y_pred = model.predict(X_test)
    labels = sorted(set(y_train) | set(y_test))
    acc = accuracy_score(y_test, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_acc = accuracy_score(y_test, baseline.predict(X_test))
    report_txt = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)

    print(f"      accuracy            {acc:.4f}   (always-guess-majority baseline: {base_acc:.4f})")
    print(f"      macro  P/R/F1       {p_m:.4f} / {r_m:.4f} / {f_m:.4f}")
    print(f"      weighted P/R/F1     {p_w:.4f} / {r_w:.4f} / {f_w:.4f}")
    print(report_txt)
    if acc > 0.995 and source == "real":
        print("      WARNING: near-perfect accuracy on real data usually means leakage "
              "(a column that encodes the label, or a random split of a time series). Investigate.")

    print("[7/8] Writing reports")
    cm = save_confusion_matrix(y_test, y_pred, labels, os.path.join(report_dir, "confusion_matrix.png"),
                               f"Confusion matrix ({tag})")
    feat_names = list(model.named_steps["prep"].get_feature_names_out())
    importances = save_feature_importance(
        feat_names, model.named_steps["rf"].feature_importances_,
        os.path.join(report_dir, "feature_importance.png"), f"Feature importance ({tag})")
    with open(os.path.join(report_dir, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Data source: {tag}\nSplit: {strategy}\n\n{report_txt}\n"
                 f"Majority-class baseline accuracy: {base_acc:.4f}\n")

    metrics = {
        "data_source": source, "split_strategy": strategy, "split_notes": notes,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "accuracy": acc, "baseline_accuracy": base_acc,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "per_class": {k: v for k, v in report_dict.items() if k in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "class_distribution": {k: int(v) for k, v in dist.items()},
        "top_features": importances,
    }
    with open(os.path.join(report_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # 12. Save --------------------------------------------------------------- #
    bundle = {
        "pipeline": model,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "categorical_values": {c: sorted(X_train[c].dropna().unique().tolist()) for c in categorical},
        "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in numeric},
        "classes": list(model.classes_),
        "label_column": label,
        "data_source": source,
        "data_files": paths,
        "config": cfg,
        "metrics": metrics,
        "report_dir": report_dir,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    out = os.path.join(model_dir, "fault_model.joblib")
    joblib.dump(bundle, out, compress=3)
    print(f"[8/8] Saved model bundle to {out}")


if __name__ == "__main__":
    main()
