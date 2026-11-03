"""
preprocess.py - Shared data loading + preprocessing for training AND inference.

Why one shared file?  The #1 bug in student ML projects is preprocessing the
training data one way and the live/API data another way.  Everything that
transforms data lives here, and the transformer itself is saved *inside* the
model pipeline, so inference cannot drift from training.

Nothing in this file assumes a specific dataset's column names. You describe
your dataset in dataset_config.json (see README).
"""
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_CONFIG = {
    "data_path": "dataset",          # a CSV file, a folder of CSVs, or a glob like "dataset/*.csv"
    "label_column": None,            # REQUIRED: the column holding the fault label
    "feature_columns": None,         # None = use every usable column except label/group/time/drop
    "drop_columns": [],              # IDs, leaky columns (e.g. a second label column), free text
    "group_column": None,            # e.g. a node / router / topology column -> group-aware split
    "time_column": None,             # e.g. a timestamp column -> chronological split
    "split_strategy": "auto",        # auto | stratified | group | chronological
    "test_size": 0.2,
    "random_state": 42,
    "sample_rows": None,             # optionally subsample huge datasets (int)
    "max_categories": 50,            # categorical columns with more unique values are dropped
}

CATEGORICAL_MISSING = "__missing__"
ID_NAME = re.compile(r"^(id|idx|index|row|row_?id|record_?id|unnamed:\s*\d+)$", re.I)


def _is_row_counter(series, n):
    """True for integer columns where every row has a different value (1, 2, 3, ...)."""
    if n <= 50:
        return False
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() < 0.95 * n or num.nunique() < 0.99 * n:
        return False
    vals = num.dropna().to_numpy()
    return bool(np.all(np.mod(vals, 1) == 0))


# --------------------------------------------------------------------------- #
# Config + loading
# --------------------------------------------------------------------------- #
def load_config(path):
    """Read a JSON config and fill in defaults for anything not given."""
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        # keys starting with "_" are comments in the JSON file
        cfg.update({k: v for k, v in user_cfg.items() if not k.startswith("_")})
    return cfg


def resolve_csv_paths(data_path):
    """Turn a file / folder / glob into a sorted list of CSV files."""
    if os.path.isdir(data_path):
        paths = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
        # never mix the synthetic test data into a real-data folder scan
        if "synthetic" not in os.path.abspath(data_path).lower():
            paths = [p for p in paths if f"{os.sep}synthetic{os.sep}" not in p]
    elif any(ch in data_path for ch in "*?["):
        paths = sorted(glob.glob(data_path, recursive=True))
    else:
        paths = [data_path] if os.path.exists(data_path) else []
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found at '{data_path}'. Download the dataset into the "
            "dataset/ folder (see README section 'Getting the dataset')."
        )
    return paths


def load_dataset(data_path, sample_rows=None, random_state=42):
    """Load one or many CSV files into a single DataFrame.

    When several files are loaded, a `source_file` column is added. It is a
    natural candidate for `group_column` (each file is often one experiment /
    capture), which prevents near-duplicate rows leaking across the split.
    """
    paths = resolve_csv_paths(data_path)
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if len(paths) > 1:
            df["source_file"] = os.path.relpath(p, os.path.dirname(paths[0]) or ".")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=int(sample_rows), random_state=random_state).sort_index()
    return df, paths


# --------------------------------------------------------------------------- #
# Feature selection + type coercion
# --------------------------------------------------------------------------- #
def select_features(df, cfg):
    """Decide which columns become numeric / categorical features.

    Returns (numeric_cols, categorical_cols, excluded) where `excluded` maps a
    column name to the reason it was not used. Printed during training so you
    can see and defend every decision.
    """
    label, group, time_col = cfg["label_column"], cfg["group_column"], cfg["time_column"]
    reserved = {c for c in (label, group, time_col) if c}
    reserved |= set(cfg.get("drop_columns") or [])

    if cfg.get("feature_columns"):
        missing = [c for c in cfg["feature_columns"] if c not in df.columns]
        if missing:
            raise KeyError(f"feature_columns not found in data: {missing}")
        candidates = [c for c in cfg["feature_columns"] if c not in reserved]
    else:
        candidates = [c for c in df.columns if c not in reserved]

    numeric, categorical, excluded = [], [], {}
    n = len(df)
    explicit = bool(cfg.get("feature_columns"))
    for col in candidates:
        s = df[col]
        if s.isna().all():
            excluded[col] = "all values missing"
            continue
        if not explicit and col == "source_file":
            excluded[col] = ("file name added by the loader - never a feature by default, "
                             "file names often encode the fault scenario (leakage)")
            continue
        if not explicit and (ID_NAME.search(col) or _is_row_counter(s, n)):
            excluded[col] = "looks like a row ID / index (would leak row order into the model)"
            continue
        if pd.api.types.is_bool_dtype(s):
            numeric.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1:
                excluded[col] = "constant value (carries no information)"
            else:
                numeric.append(col)
            continue
        # text column: is it really numbers stored as text?
        as_num = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        if non_null and as_num.notna().sum() / non_null >= 0.95:
            if as_num.nunique(dropna=True) <= 1:
                excluded[col] = "constant value (carries no information)"
            else:
                numeric.append(col)
            continue
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            excluded[col] = "constant value (carries no information)"
        elif nunique > cfg.get("max_categories", 50) or nunique / max(n, 1) > 0.5:
            excluded[col] = f"high-cardinality text ({nunique} unique) - looks like an ID/timestamp"
        else:
            categorical.append(col)
    if not numeric and not categorical:
        raise ValueError("No usable feature columns were found. Check dataset_config.json.")
    return numeric, categorical, excluded


def coerce_types(df, numeric_cols, categorical_cols):
    """Return a copy with exactly the model's columns, in a fixed order and type.

    Used identically at training time and at prediction time.
    """
    out = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        col = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        out[c] = pd.to_numeric(col, errors="coerce").astype(float)
    for c in categorical_cols:
        col = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        out[c] = col.where(col.isna(), col.astype(str)).astype(object)
    # inf values break the imputer; treat them as missing
    return out.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(numeric_cols, categorical_cols):
    """Missing-value handling + encoding, as a scikit-learn transformer.

    - numeric: missing -> median of the TRAINING data (learned in fit, so no leakage)
    - categorical: missing -> explicit "__missing__" category, then one-hot.
      Unknown categories seen at prediction time are ignored instead of crashing.
    Random Forests don't need feature scaling, so there is none.
    """
    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_MISSING)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- #
# Leakage-aware train/test split
# --------------------------------------------------------------------------- #
def _parse_time(series):
    t = pd.to_datetime(series, errors="coerce")
    if t.notna().mean() >= 0.9:
        return t
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() >= 0.9:
        return num
    raise ValueError(f"time_column '{series.name}' could not be parsed as dates or numbers")


def split_data(df, y, cfg):
    """Return (train_index, test_index, strategy_used, notes).

    - chronological: train on the past, test on the future (best for time series)
    - group: every group (node / topology / capture file) is entirely in train
      OR test, so the model is tested on network elements it never saw
    - stratified: random split keeping class proportions (only safe when rows
      are independent - usually NOT true for monitoring time series)
    """
    strategy = cfg["split_strategy"]
    test_size, seed = cfg["test_size"], cfg["random_state"]
    notes = []
    if strategy == "auto":
        if cfg.get("time_column"):
            strategy = "chronological"
        elif cfg.get("group_column"):
            strategy = "group"
        else:
            strategy = "stratified"
            notes.append(
                "No time_column or group_column configured: using a random stratified split. "
                "If rows are consecutive measurements, this OVERESTIMATES accuracy."
            )

    idx = np.arange(len(df))
    if strategy == "chronological":
        t = _parse_time(df[cfg["time_column"]])
        order = np.argsort(t.to_numpy(), kind="stable")
        cut = int(round(len(order) * (1 - test_size)))
        train_idx, test_idx = order[:cut], order[cut:]
    elif strategy == "group":
        groups = df[cfg["group_column"]].astype(str).to_numpy()
        if len(np.unique(groups)) < 2:
            raise ValueError("group split needs at least 2 distinct groups")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(idx, y, groups))
    elif strategy == "stratified":
        counts = pd.Series(y).value_counts()
        stratify = y if counts.min() >= 2 else None
        if stratify is None:
            notes.append("Some class has < 2 rows; stratification disabled.")
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=stratify
        )
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'")

    unseen = set(pd.Series(y[test_idx]).unique()) - set(pd.Series(y[train_idx]).unique())
    missing_in_test = set(pd.Series(y[train_idx]).unique()) - set(pd.Series(y[test_idx]).unique())
    if unseen:
        notes.append(f"Test set contains classes never seen in training: {sorted(unseen)}")
    if missing_in_test:
        notes.append(f"Classes absent from the test set (cannot be evaluated): {sorted(missing_in_test)}")
    for n_ in notes:
        warnings.warn(n_)
    return train_idx, test_idx, strategy, notes


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #
def records_to_frame(records, numeric_cols, categorical_cols):
    """Convert API input (dict or list of dicts) into the exact training layout."""
    if isinstance(records, dict):
        records = [records]
    raw = pd.DataFrame.from_records(records)
    return coerce_types(raw, numeric_cols, categorical_cols)
