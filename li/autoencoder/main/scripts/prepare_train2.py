#!/usr/bin/env python3
"""
Build aligned pickle datasets for each ECG lead from train2.json.

IMPORTANT PROPERTY:
-------------------
Row i in ALL generated files corresponds to THE SAME patient.
If at least one lead fails for a patient, the patient is dropped entirely.

For each lead (I, II, V1–V6) the script creates:
  - train2<lead>.pkl from names/train2.json

Each sample consists of 5 consecutive cardiac periods from ONE lead,
filtered and normalized per-period. Intended for ensemble model training.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm
import yaml


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LEADS = [
    "ECG I",
    "ECG II",
    "ECG V1",
    "ECG V2",
    "ECG V3",
    "ECG V4",
    "ECG V5",
    "ECG V6",
]

NUM_PERIODS = 5


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_edf_record(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[float]]:
    try:
        reader = pyedflib.EdfReader(str(file_path))
        try:
            signals = np.vstack([reader.readSignal(i) for i in range(reader.signals_in_file)])
            labels = reader.getSignalLabels()
            sfreq = reader.getSampleFrequency(0)
        finally:
            reader._close()
            del reader
        return signals, labels, sfreq
    except Exception:
        return None, None, None


# -----------------------------------------------------------------------------
# Signal processing
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


def process_record_single_lead(
    record_path: Path,
    lead: str,
    num_periods: int,
) -> Optional[List[float]]:
    signals, labels, sfreq = read_edf_record(record_path)
    if signals is None or lead not in labels:
        return None

    idx = labels.index(lead)
    x = signals[idx]

    # Bandpass filtering
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
# Dataset building (PATIENT-ALIGNED)
# -----------------------------------------------------------------------------

def collect_record_paths(base: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for i in range(1, 12):
        folder = base / f"{i}/{i}"
        if not folder.exists():
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".edf"):
                    paths[Path(f).stem] = Path(root) / f
    return paths


def load_lef_names(base: Path) -> set[str]:
    df = pd.read_csv(base / "lef_names.csv", header=None)
    return set(df[0].apply(lambda x: Path(str(x)).stem))


def build_split(
    *,
    split_name: str,
    json_path: Path,
    record_paths: Dict[str, Path],
    lef_names: set[str],
    out_dir: Path,
) -> None:

    with open(json_path, "r") as f:
        data = json.load(f)

    # FIXED, deterministic patient order
    patient_list = sorted(
        Path(v).stem for k, v in data.items() if str(k).isdigit()
    )

    signals_by_lead = {lead: [] for lead in LEADS}
    labels: List[int] = []
    patient_ids: List[str] = []

    skipped = 0

    for name in tqdm(patient_list, desc=f"Building {split_name}"):
        if name not in record_paths:
            skipped += 1
            continue

        per_lead_signals: Dict[str, List[float]] = {}

        for lead in LEADS:
            sig = process_record_single_lead(
                record_paths[name], lead, NUM_PERIODS
            )
            if sig is None:
                break
            per_lead_signals[lead] = sig

        # Drop patient entirely if ANY lead failed
        if len(per_lead_signals) != len(LEADS):
            skipped += 1
            continue

        for lead in LEADS:
            signals_by_lead[lead].append(per_lead_signals[lead])

        labels.append(1 if name in lef_names else 0)
        patient_ids.append(name)

    # Save aligned datasets
    for lead in LEADS:
        df = pd.DataFrame({
            "patient_id": patient_ids,
            "signal": signals_by_lead[lead],
            "label": labels,
        })

        lead_suffix = lead.split()[-1].lower()
        out_path = out_dir / f"{split_name}{lead_suffix}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(df, f)

        print(f"Saved {out_path} | {len(df)} patients")

    print(f"Finished {split_name}: kept {len(patient_ids)}, skipped {skipped}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # Read all paths from YAML config instead of hardcoding
    with open("configs/training_params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    edf_root = Path(cfg["edf_root"])
    names_root = Path(cfg["names_root"])
    out_dir = Path(cfg["datasets12_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)

    record_paths = collect_record_paths(edf_root)
    lef_names = load_lef_names(edf_root)

    build_split(
        split_name="train2",
        json_path=names_root / "train2.json",
        record_paths=record_paths,
        lef_names=lef_names,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

