#!/usr/bin/env python
"""
Описательная статистика по датасету в формате pandas describe().

Размеченные: записи из lef_names.csv (LFV, label=1).
Неразмеченные: остальные EDF без метки LFV.

Пример:
  ECGApp/venv/bin/python scripts/dataset_descriptive_stats.py
  ECGApp/venv/bin/python scripts/dataset_descriptive_stats.py --max-unlabeled 500
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_io import (  # noqa: E402
    TARGET_CHANNELS,
    discover_edf_paths,
    load_lef_names,
    process_ecg_data,
    read_edf_path,
)

DATASETS_ROOT = Path("/Users/moomintroll/Desktop/Научка_мага/datasets")
OUTPUT_DIR = REPO_ROOT / "dataset_stats"
DEFAULT_BEAT_INDEX = 2

# Как в примере пользователя: v1..v6 для грудных отведений
LEAD_COLUMNS = ["v1", "v2", "v3", "v4", "v5", "v6"]
LEAD_MAP = {
    "v1": "ECG V1",
    "v2": "ECG V2",
    "v3": "ECG V3",
    "v4": "ECG V4",
    "v5": "ECG V5",
    "v6": "ECG V6",
}


def period_features(lead_signal):
    signal = np.asarray(lead_signal, dtype=float)
    if signal.size == 0:
        return {k: np.nan for k in ["mean", "std", "min", "max", "ptp", "rms"]}
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "ptp": float(np.ptp(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
    }


def record_feature_row(edf_path, beat_index):
    signals, channel_labels, sfreq = read_edf_path(edf_path)
    _, lead_periods = process_ecg_data(
        signals, channel_labels, sfreq, beat_index=beat_index
    )
    row = {"sfreq": sfreq, "period_len": len(next(iter(lead_periods.values())))}
    for col, ch in LEAD_MAP.items():
        feats = period_features(lead_periods[ch])
        for name, value in feats.items():
            row[f"{col}_{name}"] = value
    return row


def build_feature_table(paths, names, beat_index):
    rows = []
    for name in names:
        try:
            row = record_feature_row(paths[name], beat_index)
            row["record_id"] = name
            rows.append(row)
        except Exception as exc:
            print(f"  skip {name}: {exc}")
    return pd.DataFrame(rows)


def zscore_columns(df, columns):
    out = df[columns].copy()
    for col in columns:
        vals = out[col].astype(float)
        std = vals.std(ddof=0)
        if std > 1e-12:
            out[col] = (vals - vals.mean()) / std
        else:
            out[col] = 0.0
    return out


def format_describe_table(df, columns, zscore=False):
    use_df = zscore_columns(df, columns) if zscore else df[columns]
    desc = use_df.describe().T
    desc.index.name = None
    return desc


def save_describe_outputs(desc, prefix, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    desc.round(6).to_csv(out_dir / f"{prefix}_describe.csv")
    with open(out_dir / f"{prefix}_describe.txt", "w", encoding="utf-8") as f:
        f.write(desc.round(6).to_string())
        f.write("\n")


def print_block(title, desc):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(desc.round(6).to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="describe() по размеченным и неразмеченным EDF")
    parser.add_argument("--beat-index", type=int, default=DEFAULT_BEAT_INDEX)
    parser.add_argument(
        "--max-unlabeled",
        type=int,
        default=500,
        help="Сколько неразмеченных записей обработать (0 = все)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        choices=["mean", "std", "ptp", "rms"],
        default="mean",
        help="Какой скаляр по периоду отведения выводить в v1..v6",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Перед describe() стандартизировать столбцы (mean~0, std~1)",
    )
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    metric = args.metric

    paths = discover_edf_paths(DATASETS_ROOT)
    lef_names = load_lef_names(DATASETS_ROOT / "lef_names.csv")

    labeled_names = sorted(lef_names & set(paths.keys()))
    unlabeled_names = sorted(set(paths.keys()) - lef_names)

    if args.max_unlabeled > 0 and len(unlabeled_names) > args.max_unlabeled:
        rng = random.Random(args.seed)
        unlabeled_names = sorted(rng.sample(unlabeled_names, args.max_unlabeled))

    print(f"Размеченные (LFV): {len(labeled_names)}")
    print(f"Неразмеченные: {len(unlabeled_names)}")

    print("\nСбор признаков по размеченным...")
    labeled_df = build_feature_table(paths, labeled_names, args.beat_index)
    print("Сбор признаков по неразмеченным...")
    unlabeled_df = build_feature_table(paths, unlabeled_names, args.beat_index)

    columns = [f"{col}_{metric}" for col in LEAD_COLUMNS]
    rename = {f"{col}_{metric}": col for col in LEAD_COLUMNS}

    labeled_view = labeled_df.rename(columns=rename)
    unlabeled_view = unlabeled_df.rename(columns=rename)

    labeled_desc = format_describe_table(labeled_view, LEAD_COLUMNS, zscore=args.zscore)
    unlabeled_desc = format_describe_table(unlabeled_view, LEAD_COLUMNS, zscore=args.zscore)

    save_describe_outputs(labeled_desc, "labeled", out_dir)
    save_describe_outputs(unlabeled_desc, "unlabeled", out_dir)

    meta = pd.DataFrame(
        [
            {
                "group": "labeled",
                "n_records": len(labeled_df),
                "metric": metric,
                "beat_index": args.beat_index,
                "zscore_before_describe": args.zscore,
            },
            {
                "group": "unlabeled",
                "n_records": len(unlabeled_df),
                "metric": metric,
                "beat_index": args.beat_index,
                "zscore_before_describe": args.zscore,
            },
        ]
    )
    meta.to_csv(out_dir / "meta.csv", index=False)

    print_block(
        f"Размеченные данные (n={len(labeled_df)}), метрика={metric}, v1..v6",
        labeled_desc,
    )
    print_block(
        f"Неразмеченные данные (n={len(unlabeled_df)}), метрика={metric}, v1..v6",
        unlabeled_desc,
    )

    if not args.zscore:
        print_block(
            "Размеченные (z-score столбцов перед describe, как в примере)",
            format_describe_table(labeled_view, LEAD_COLUMNS, zscore=True),
        )
        print_block(
            "Неразмеченные (z-score столбцов перед describe, как в примере)",
            format_describe_table(unlabeled_view, LEAD_COLUMNS, zscore=True),
        )
        save_describe_outputs(
            format_describe_table(labeled_view, LEAD_COLUMNS, zscore=True),
            "labeled_zscore",
            out_dir,
        )
        save_describe_outputs(
            format_describe_table(unlabeled_view, LEAD_COLUMNS, zscore=True),
            "unlabeled_zscore",
            out_dir,
        )

    print(f"\nСохранено в {out_dir}")


if __name__ == "__main__":
    main()
