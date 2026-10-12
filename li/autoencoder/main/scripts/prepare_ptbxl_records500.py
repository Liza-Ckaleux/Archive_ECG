#!/usr/bin/env python3
"""
Integrate PTB-XL `records500` data into existing Autoencoder train pickles.

This script:
  1. Reads PTB-XL metadata (`ptbxl_database.csv`)
  2. Selects *normal* ECGs (scp_labels containing "NORM")
  3. Loads signals from `records500` via `wfdb`
  4. Extracts several consecutive cardiac periods from the chosen lead
  5. Normalizes & flattens them (same idea as `prepare_train_zero.py`)
  6. Appends these samples to your existing train pickle and saves the result

Resulting DataFrame structure matches existing ones:
  - columns: ["patient_id", "signal", "label"]
  - `signal` is a list[float], `label` is 0 for normal beats
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import wfdb  # type: ignore
from tqdm import tqdm
import yaml


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Which PTB-XL channel names correspond to which lead suffix in your pickles
PTBXL_LEAD_MAP: Dict[str, str] = {
    "i": "I",
    "ii": "II",
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v5": "V5",
    "v6": "V6",
}

NUM_PERIODS = 5


# -----------------------------------------------------------------------------
# Signal processing (copied conceptually from `prepare_train_zero.py`)
# -----------------------------------------------------------------------------

def extract_periods(signal: np.ndarray, sfreq: float, num_periods: int) -> List[np.ndarray]:
    peaks, _ = find_peaks(signal, distance=int(0.6 * sfreq))
    if len(peaks) < num_periods + 1:
        raise ValueError("Not enough peaks")

    periods: List[np.ndarray] = []
    for i in range(num_periods):
        start, end = peaks[i], peaks[i + 1]
        half = (end - start) // 2
        periods.append(signal[end - half : end + half])
    return periods


def normalize_period(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def process_ptbxl_record_single_lead(
    base_path: Path,
    rel_wfdb_path: str,
    target_lead: str,
    num_periods: int,
) -> Optional[List[float]]:
    """
    Load a PTB-XL WFDB record and extract one processed lead.

    Parameters
    ----------
    base_path : Path
        Root path of PTB-XL dataset (folder that contains `records500` and `ptbxl_database.csv`).
    rel_wfdb_path : str
        Relative path stored in ptbxl_database.csv columns `filename_lr`/`filename_hr`
        (e.g. 'records500/00000/00001_hr').
    target_lead : str
        Channel name in PTB-XL, e.g. 'I', 'II', 'V1', ...
    """
    record_path = base_path / rel_wfdb_path
    # wfdb expects path *without* extension
    record_path = record_path.with_suffix("")

    try:
        sig, fields = wfdb.rdsamp(str(record_path))
    except Exception:
        return None

    sig_names = fields.get("sig_name", [])
    if target_lead not in sig_names:
        return None

    idx = sig_names.index(target_lead)
    x = sig[:, idx]
    sfreq = float(fields.get("fs", 500.0))

    # Bandpass filtering (same range as in prepare_train_zero.py)
    nyq = 0.5 * sfreq
    b, a = butter(1, [0.6 / nyq, 40 / nyq], btype="bandpass")
    x = filtfilt(b, a, x)

    try:
        periods = extract_periods(x, sfreq, num_periods)
    except Exception:
        return None

    periods = [normalize_period(p) for p in periods]
    return np.hstack(periods).astype(np.float32).tolist()


# -----------------------------------------------------------------------------
# PTB-XL helpers
# -----------------------------------------------------------------------------

def load_ptbxl_metadata(base_path: Path) -> pd.DataFrame:
    """
    Load PTB-XL metadata and keep only normal ECGs.

    In the Kaggle PTB-XL dump the column with diagnostic codes is called
    `scp_codes` (not `scp_labels`), and it stores a dict-like string.
    """
    csv_path = base_path / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    col_name = "scp_codes"
    if col_name not in df.columns:
        raise KeyError(
            f"Expected column '{col_name}' in ptbxl_database.csv, "
            f"got columns: {list(df.columns)}"
        )

    # Parse scp_codes JSON-like column
    df["scp_parsed"] = df[col_name].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else {}
    )

    # Keep only normal ECGs (contains key 'NORM')
    df["is_normal"] = df["scp_parsed"].apply(lambda d: int("NORM" in d.keys()))
    df = df[df["is_normal"] == 1].copy()
    return df


def build_ptbxl_dataset_for_lead(
    base_path: Path,
    lead_suffix: str,
    use_high_res: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns [patient_id, signal, label] for one lead.

    lead_suffix: 'i', 'ii', 'v1', ..., matching your existing pickle filenames.
    """
    if lead_suffix not in PTBXL_LEAD_MAP:
        raise ValueError(f"Unsupported lead_suffix={lead_suffix}. Use one of {list(PTBXL_LEAD_MAP.keys())}")

    target_lead_name = PTBXL_LEAD_MAP[lead_suffix]

    df_meta = load_ptbxl_metadata(base_path)
    filename_col = "filename_hr" if use_high_res else "filename_lr"
    if filename_col not in df_meta.columns:
        raise KeyError(f"Column {filename_col} not found in ptbxl_database.csv")

    signals: List[List[float]] = []
    labels: List[int] = []
    patient_ids: List[str] = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=f"PTB-XL {target_lead_name}"):
        rel_path = row[filename_col]
        patient_id = str(row["ecg_id"])

        sig = process_ptbxl_record_single_lead(
            base_path=base_path,
            rel_wfdb_path=rel_path,
            target_lead=target_lead_name,
            num_periods=NUM_PERIODS,
        )
        if sig is None:
            continue

        signals.append(sig)
        labels.append(0)  # normal class
        patient_ids.append(patient_id)

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "signal": signals,
            "label": labels,
        }
    )
    return df


def append_to_existing_train(existing_train_path: Path, extra_df: pd.DataFrame, out_path: Path) -> None:
    if existing_train_path.exists():
        base_df: pd.DataFrame = pd.read_pickle(existing_train_path)
        combined = pd.concat([base_df, extra_df], ignore_index=True)
    else:
        combined = extra_df

    combined.to_pickle(out_path)
    print(f"Saved combined train to {out_path} | {len(combined)} samples")


def main() -> None:
    # Paths from YAML, only lead/low_res остаются параметрами CLI
    with open("configs/training_params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lead",
        type=str,
        default="v1",
        help="Lead suffix to process: one of i, ii, v1, v2, v3, v4, v5, v6. "
             "This corresponds to train<lead>.pkl in your datasets folder.",
    )
    parser.add_argument(
        "--low_res",
        action="store_true",
        help="Use filename_lr instead of filename_hr from ptbxl_database.csv.",
    )
    args = parser.parse_args()

    ptbxl_root = Path(cfg["ptbxl_root"])
    datasets_dir = Path(cfg["datasets12_dir"])

    lead_suffix = args.lead.lower()

    extra_df = build_ptbxl_dataset_for_lead(
        base_path=ptbxl_root,
        lead_suffix=lead_suffix,
        use_high_res=not args.low_res,
    )

    existing_train = datasets_dir / f"train{lead_suffix}.pkl"
    out_path = existing_train  # overwrite / extend in place

    append_to_existing_train(existing_train, extra_df, out_path)


if __name__ == "__main__":
    main()


