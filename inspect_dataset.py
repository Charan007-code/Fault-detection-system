"""
inspect_dataset.py - STEP 1: look at the real data before training anything.

    python inspect_dataset.py dataset/            # a folder of CSVs
    python inspect_dataset.py dataset/file.csv    # a single CSV

It prints every column (type, % missing, unique values, examples) and
*suggests* which column might be the label, the time column, and a grouping
column. It never picks for you - you confirm the choice in dataset_config.json,
because guessing a label column is exactly how fake results happen.

Output is also saved to reports/dataset_inspection.json.
"""
import json
import os
import re
import sys

import pandas as pd

from preprocess import ID_NAME, _is_row_counter, load_dataset

LABEL_HINTS = re.compile(r"label|fault|class|status|state|event|anomal|target|type|failure", re.I)
TIME_HINTS = re.compile(r"time|date|timestamp|epoch|^ts$", re.I)
GROUP_HINTS = re.compile(r"node|router|device|host|interface|link|topolog|scenario|experiment|source_file|run", re.I)


def inspect(path):
    df, paths = load_dataset(path)
    n = len(df)
    print(f"\nLoaded {len(paths)} file(s), {n:,} rows, {df.shape[1]} columns")
    for p in paths[:20]:
        print("   -", p)
    if len(paths) > 20:
        print(f"   ... and {len(paths) - 20} more")

    report = {"files": paths, "rows": n, "columns": {}}
    label_cand, time_cand, group_cand, id_like = [], [], [], []

    print(f"\n{'column':40s} {'dtype':10s} {'missing%':>8s} {'unique':>8s}  examples")
    print("-" * 110)
    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        miss = float(s.isna().mean() * 100)
        examples = [str(v)[:25] for v in s.dropna().unique()[:4]]
        info = {"dtype": str(s.dtype), "missing_pct": round(miss, 2), "unique": nunique,
                "examples": examples}
        if nunique <= 30:
            info["value_counts"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}
        report["columns"][col] = info
        print(f"{col[:40]:40s} {str(s.dtype)[:10]:10s} {miss:8.1f} {nunique:8d}  {', '.join(examples)}")

        if 2 <= nunique <= 30 and (LABEL_HINTS.search(col) or not pd.api.types.is_numeric_dtype(s)):
            label_cand.append(col)
        if TIME_HINTS.search(col):
            time_cand.append(col)
        if GROUP_HINTS.search(col) and 2 <= nunique < n * 0.5:
            group_cand.append(col)
        # same rule as training: an ID is a named id/index column, an integer
        # row counter, or a TEXT column unique per row. Continuous measurements
        # (e.g. rtt 20.13, 20.71 ...) are also unique per row but are NOT IDs.
        is_text = not pd.api.types.is_numeric_dtype(s)
        if (ID_NAME.search(col) or _is_row_counter(s, n) or (is_text and nunique == n and n > 50)) \
                and not TIME_HINTS.search(col):
            id_like.append(col)

    report["suggestions"] = {
        "label_candidates": label_cand, "time_candidates": time_cand,
        "group_candidates": group_cand, "id_like_columns": id_like,
    }
    print("\nSUGGESTIONS (verify against the dataset's README / paper!)")
    print("  possible label columns :", label_cand or "none found")
    for c in label_cand:
        print(f"      {c}: {report['columns'][c].get('value_counts')}")
    print("  possible time columns  :", time_cand or "none found")
    print("  possible group columns :", group_cand or "none found")
    print("  ID-like (drop these)   :", id_like or "none found")
    print("\nWARNING: if there is more than one label-like column (e.g. 'fault_type' and"
          "\n'is_faulty'), put the extra ones in drop_columns - otherwise the model will"
          "\n'cheat' by reading the answer from another column (data leakage).")

    os.makedirs("reports", exist_ok=True)
    with open("reports/dataset_inspection.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in report.items() if k != "_df"}, fh, indent=2, default=str)
    print("\nSaved full report to reports/dataset_inspection.json")
    print("Next: edit dataset_config.json (or run  python automate.py configure), then  python train_model.py")
    report["_df"] = df  # in-memory only, used by automate.py
    return report


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else "dataset")
