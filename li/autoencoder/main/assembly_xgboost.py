#!/usr/bin/env python3
"""
Ensemble model that combines features from all 8 ECG leads using XGBoost.

Pipeline:
1. For each lead (i, ii, v1-v6):
   - Load trained autoencoder
   - Extract features (reconstruction error MSE + latent vector)
2. Combine all features from all leads into unified feature vector
3. Train XGBoost on combined features (val split from test)
4. Evaluate on test split (50/50 split of original test data)
5. Generate multilead raw ECG figures for correctly diagnosed patients

This approach uses raw AE features directly with XGBoost classifier.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

# Fix OpenMP / segfault on macOS: один поток до загрузки XGBoost
os.environ["OMP_NUM_THREADS"] = "1"

# Try to find libomp.dylib in sklearn's .dylibs directory
try:
    import sklearn
    sklearn_path = Path(sklearn.__file__).parent
    libomp_path = sklearn_path / '.dylibs' / 'libomp.dylib'
    if libomp_path.exists():
        libomp_dir = str(libomp_path.parent)
        # Add to DYLD_LIBRARY_PATH (though it may not work with SIP, we try)
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        if libomp_dir not in current_dyld:
            os.environ['DYLD_LIBRARY_PATH'] = f'{libomp_dir}:{current_dyld}'.rstrip(':')
except Exception:
    pass

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Настройка шрифтов для всех графиков
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20

from models.autoencoder import AutoencoderConfig, ECGAutoencoder


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LEADS = ["i", "ii", "v1", "v2", "v3", "v4", "v5", "v6"]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ECGLeadDataset(Dataset):
    """One record = one lead, signal is np.array or list."""

    def __init__(self, df: pd.DataFrame, target_length: int | None = None):
        self.df = df.reset_index(drop=True)
        self.labels = df["label"].astype(int).to_numpy()
        self.signals = [np.asarray(s) for s in df["signal"]]

        lengths = [len(s) for s in self.signals]
        self.target_length = min(lengths) if target_length is None else target_length

    def _fix_length(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.target_length:
            return x[: self.target_length]
        if len(x) < self.target_length:
            return np.pad(x, (0, self.target_length - len(x)))
        return x

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        sig = self._fix_length(self.signals[idx])
        label = self.labels[idx]
        return (
            torch.tensor(sig, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def collect_features(
    model: ECGAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      features: [N, 1 + latent_dim] = [MSE, latent...]
      reconstructions: [N, signal_length]
      labels: [N]
    """
    model.eval()
    all_features = []
    all_reconstructions = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass - model returns dict with "reconstruction" and "latent"
            out = model(signals)
            reconstructions = out["reconstruction"]
            latent = out["latent"]

            # Calculate MSE reconstruction error per sample
            mse_per_sample = torch.mean((signals - reconstructions) ** 2, dim=1)

            # Combine MSE and latent
            features = torch.cat([mse_per_sample.unsqueeze(1), latent], dim=1)

            all_features.append(features.cpu().numpy())
            all_reconstructions.append(reconstructions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    features = np.vstack(all_features)
    reconstructions = np.vstack(all_reconstructions)
    labels = np.concatenate(all_labels)

    return features, reconstructions, labels


def load_ensemble_features(
    train2_dir: Path,
    test_dir: Path,
    artifacts_base_dir: Path,
    device: torch.device,
    batch_size: int,
    latent_dim: int,
    seed: int,
    val_fraction: float = 0.33,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and combine features from all leads.
    
    Data split structure:
    - train2: Additional train data (both classes) - used entirely for training ensemble model
    - test: Test data (both classes) - split into val/test (val for validation, test for final evaluation)
    
    Returns:
      X_train: [N_train, n_features] - Features from train2 (all leads combined)
      X_val: [N_val, n_features] - Features from test (val split, all leads combined)
      X_test: [N_test, n_features] - Features from test (test split, all leads combined)
      y_train: [N_train] - Labels from train2
      y_val: [N_val] - Labels from test (val split)
      y_test: [N_test] - Labels from test (test split)
    """
    log = logging.getLogger("assembly")
    
    all_train_features = []
    all_val_features = []
    all_test_features = []
    y_train = None
    y_val = None
    y_test = None
    
    for lead in LEADS:
        log.info(f"Processing lead: {lead}")
        
        artifacts_dir = artifacts_base_dir / f"artifacts_{lead}"
        if not artifacts_dir.exists():
            log.warning(f"Artifacts directory not found: {artifacts_dir}, skipping lead {lead}")
            continue
        
        # Load train2 data (entirely for training ensemble model)
        train2_df, _ = load_lead_data(train2_dir, lead, "train2")
        
        # Load test data and split into val and test
        test_df, _ = load_lead_data(test_dir, lead, "test")
        
        # Split test into val and test
        test_full = ECGLeadDataset(test_df)
        val_size = int(len(test_full) * val_fraction)
        test_size = len(test_full) - val_size
        val_subset, test_subset = random_split(
            test_full,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        
        # Get indices from subsets
        val_indices = val_subset.indices
        test_indices = test_subset.indices
        
        val_df_split = test_df.iloc[val_indices].reset_index(drop=True)
        test_df_split = test_df.iloc[test_indices].reset_index(drop=True)
        
        # Extract features from train2 (entirely for training)
        X_train_lead, _, y_train_lead, _ = extract_lead_features(
            lead=lead,
            artifacts_dir=artifacts_dir,
            val_df=train2_df,
            test_df=train2_df,
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
        )
        
        # Extract features from test (split into val and test)
        X_val_lead, X_test_lead, y_val_lead, y_test_lead = extract_lead_features(
            lead=lead,
            artifacts_dir=artifacts_dir,
            val_df=val_df_split,
            test_df=test_df_split,
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
        )
        
        all_train_features.append(X_train_lead)
        all_val_features.append(X_val_lead)
        all_test_features.append(X_test_lead)
        
        if y_train is None:
            y_train = y_train_lead
        else:
            if not np.array_equal(y_train, y_train_lead):
                log.warning(f"Train label mismatch for lead {lead}, using first lead's labels")
        
        if y_val is None:
            y_val = y_val_lead
        else:
            if not np.array_equal(y_val, y_val_lead):
                log.warning(f"Val label mismatch for lead {lead}, using first lead's labels")
        
        if y_test is None:
            y_test = y_test_lead
        else:
            if not np.array_equal(y_test, y_test_lead):
                log.warning(f"Test label mismatch for lead {lead}, using first lead's labels")
    
    if len(all_train_features) == 0:
        raise ValueError("No leads processed successfully!")
    
    # Concatenate features from all leads
    # Each lead contributes: [MSE, latent_1, ..., latent_N]
    # Total: 8 leads * (1 + latent_dim) features
    X_train = np.hstack(all_train_features)
    X_val = np.hstack(all_val_features)
    X_test = np.hstack(all_test_features)
    
    log.info(f"Combined feature shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    log.info(f"Feature structure: {len(LEADS)} leads × (1 MSE + {latent_dim} latent) = {X_train.shape[1]} features")
    
    assert y_train is not None and y_val is not None and y_test is not None, "Labels not collected properly"
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# Train Classifier (XGBoost)
# -----------------------------------------------------------------------------

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 1500,
    learning_rate: float = 0.01,
    max_depth: int = 16,
    seed: int = 42,
) -> Tuple[object, Dict]:
    """
    Train XGBoost classifier on combined features from all leads.
    
    Returns:
      model: Trained model
      metrics: Dictionary with training metrics
    """
    log = logging.getLogger("assembly")
    
    # Не менять DYLD_LIBRARY_PATH — это может вызвать загрузку несовместимых библиотек и segfault.
    
    y_train_int = y_train.astype(int)
    y_val_int = y_val.astype(int)
    
    log.info("Training XGBoost classifier...")
    
    # Additional data validation
    log.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    log.info(f"y_train shape: {y_train_int.shape}, dtype: {y_train_int.dtype}, unique: {np.unique(y_train_int)}")
    log.info(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
    log.info(f"y_val shape: {y_val_int.shape}, dtype: {y_val_int.dtype}, unique: {np.unique(y_val_int)}")
    
    # Ensure contiguous arrays
    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        random_state=seed,
        use_label_encoder=False,
        tree_method="hist",   # hist часто стабильнее на macOS (меньше риска segfault)
        n_jobs=1,
        verbosity=0,  # Suppress warnings
    )
    
    evals_result = {}
    try:
        model.fit(
            X_train,
            y_train_int,
            eval_set=[(X_train, y_train_int), (X_val, y_val_int)],
            verbose=False,
        )
    except Exception as e:
        log.error(f"XGBoost fit error: {e}")
        log.error(f"X_train stats: min={X_train.min()}, max={X_train.max()}, mean={X_train.mean()}, std={X_train.std()}")
        log.error(f"y_train stats: {np.bincount(y_train_int)}")
        raise
    
    # Get evaluation results
    evals_result = model.evals_result()
    
    # Get validation predictions
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val_int, val_probs)
    
    # Extract training history
    train_loss = evals_result["validation_0"]["logloss"]
    val_loss = evals_result["validation_1"]["logloss"]
    train_auc = evals_result["validation_0"]["auc"]
    val_auc_history = evals_result["validation_1"]["auc"]
    
    metrics = {
        "val_auc": val_auc,
        "feature_importance": model.feature_importances_,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_auc": train_auc,
        "val_auc_history": val_auc_history,
    }
    
    log.info(f"Training completed. Validation AUC: {val_auc:.4f}")
    
    return model, metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    with open("configs/training_params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Все параметры читаются из YAML, аргументы командной строки опциональны для переопределения
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset/pathology name, e.g. ph, lef (overrides YAML, saves to artifacts_ensemble_xgb_{name})")
    parser.add_argument("--dataset_dir", type=Path, default=None,
                        help="Path to Ph-style folder with X_train.pkl, X_test.pkl (overrides YAML)")
    parser.add_argument("--train2_dir", type=Path, default=None,
                        help="Directory with train2 data (overrides YAML)")
    parser.add_argument("--test_dir", type=Path, default=None,
                        help="Directory with test data (overrides YAML)")
    parser.add_argument("--artifacts_base_dir", type=Path, default=None,
                        help="Base directory with autoencoder artifacts (overrides YAML)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides YAML)")
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Latent dimension (overrides YAML)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides YAML)")
    parser.add_argument("--val_fraction", type=float, default=None,
                        help="Validation fraction (overrides YAML)")
    parser.add_argument("--n_estimators", type=int, default=None,
                        help="Number of XGBoost estimators (overrides YAML)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="XGBoost learning rate (overrides YAML)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="XGBoost max depth (overrides YAML)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (overrides YAML)")
    parser.add_argument("--generate_raw_ecg_figures", type=lambda x: x.lower() == "true", default=None,
                        help="Generate multilead raw ECG figures (overrides YAML)")
    parser.add_argument("--n_periods", type=int, default=None,
                        help="Number of cardiac periods in raw ECG figures (overrides YAML)")
    args = parser.parse_args()
    
    # Используем значения из аргументов, если указаны, иначе из YAML
    dataset_name = (args.dataset_name or cfg.get("dataset_name") or "").strip()
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else (Path(cfg.get("dataset_dir")) if cfg.get("dataset_dir") else None)
    use_ph_style = bool(
        dataset_name
        and dataset_dir
        and (dataset_dir / "X_train.pkl").exists()
    )

    if use_ph_style:
        train2_dir = test_dir = dataset_dir
        val_fraction = args.val_fraction if args.val_fraction is not None else 0.125
        out_dir = args.out if args.out is not None else Path(f"artifacts_ensemble_xgb_{dataset_name}")
    else:
        train2_dir = args.train2_dir if args.train2_dir is not None else Path(cfg.get("train2_dir", cfg.get("datasets12_dir")))
        test_dir = args.test_dir if args.test_dir is not None else Path(cfg.get("test_dir", cfg.get("datasets12_dir")))
        val_fraction = args.val_fraction if args.val_fraction is not None else cfg.get("val_fraction", 0.33)
        out_dir = args.out if args.out is not None else Path(cfg.get("ensemble_output", "artifacts_ensemble_xgb"))

    artifacts_base_dir = args.artifacts_base_dir if args.artifacts_base_dir is not None else Path(cfg.get("artifacts_base_dir", "."))
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    latent_dim = args.latent_dim if args.latent_dim is not None else cfg.get("latent_dim", 16)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    n_estimators = args.n_estimators if args.n_estimators is not None else cfg.get("n_estimators", 400)
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg.get("learning_rate", 0.005)
    max_depth = args.max_depth if args.max_depth is not None else cfg.get("max_depth", 16)
    generate_raw_ecg_figures = (
        args.generate_raw_ecg_figures
        if args.generate_raw_ecg_figures is not None
        else cfg.get("generate_raw_ecg_figures", cfg.get("generate_heatmaps", True))
    )
    n_periods = args.n_periods if args.n_periods is not None else cfg.get("n_periods", 5)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("assembly")
    
    log.info("=" * 60)
    log.info("Configuration:")
    log.info(f"  dataset_name: {dataset_name or '(per-lead)'}")
    log.info(f"  use_ph_style: {use_ph_style}")
    log.info(f"  train2_dir: {train2_dir}")
    log.info(f"  test_dir: {test_dir}")
    log.info(f"  artifacts_base_dir: {artifacts_base_dir}")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  latent_dim: {latent_dim}")
    log.info(f"  seed: {seed}")
    log.info(f"  val_fraction: {val_fraction}")
    log.info(f"  n_estimators: {n_estimators}")
    log.info(f"  learning_rate: {learning_rate}")
    log.info(f"  max_depth: {max_depth}")
    log.info(f"  output_dir: {out_dir}")
    log.info(f"  generate_raw_ecg_figures: {generate_raw_ecg_figures}")
    log.info(f"  n_periods: {n_periods}")
    log.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Load ensemble features (MSE + latent from each lead)
    log.info("Loading features from all leads (MSE + latent)...")
    if use_ph_style:
        X_train, X_val, X_test, y_train, y_val, y_test = load_ensemble_features_from_ph(
            dataset_dir=dataset_dir,
            artifacts_base_dir=artifacts_base_dir,
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
            seed=seed,
            val_fraction=val_fraction,
        )
        log.info("=" * 60)
        log.info("Data Split Summary (Ph-style: val from train 0.125, test full):")
        log.info(f"  train: {X_train.shape[0]} samples - for training XGBoost")
        log.info(f"  val:   {X_val.shape[0]} samples - from train (validation)")
        log.info(f"  test:  {X_test.shape[0]} samples - full test set")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_ensemble_features(
            train2_dir=train2_dir,
            test_dir=test_dir,
            artifacts_base_dir=artifacts_base_dir,
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
            seed=seed,
            val_fraction=val_fraction,
        )
        log.info("=" * 60)
        log.info("Data Split Summary:")
        log.info(f"  train (full train2): {X_train.shape[0]} samples - used for training XGBoost")
        log.info(f"  val (from test):      {X_val.shape[0]} samples - used for validation")
        log.info(f"  test (from test):     {X_test.shape[0]} samples - used for final evaluation")
    log.info(f"  Features per sample: {X_train.shape[1]} (8 leads × (1 MSE + {latent_dim} latent))")
    log.info("=" * 60)
    
    # Validate and clean data before training
    log.info("Validating and cleaning data...")
    
    # Check for NaN and inf
    train_nan_count = np.isnan(X_train).sum()
    train_inf_count = np.isinf(X_train).sum()
    val_nan_count = np.isnan(X_val).sum()
    val_inf_count = np.isinf(X_val).sum()
    test_nan_count = np.isnan(X_test).sum()
    test_inf_count = np.isinf(X_test).sum()
    
    log.info(f"NaN counts - Train: {train_nan_count}, Val: {val_nan_count}, Test: {test_nan_count}")
    log.info(f"Inf counts - Train: {train_inf_count}, Val: {val_inf_count}, Test: {test_inf_count}")
    
    # Clean data: replace NaN and inf with finite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check data ranges
    log.info(f"Train data range: [{X_train.min():.6f}, {X_train.max():.6f}]")
    log.info(f"Val data range: [{X_val.min():.6f}, {X_val.max():.6f}]")
    log.info(f"Test data range: [{X_test.min():.6f}, {X_test.max():.6f}]")
    
    # Ensure data types are correct
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    
    # Check label distribution
    log.info(f"Train labels - Normal: {(y_train == 0).sum()}, Abnormal: {(y_train == 1).sum()}")
    log.info(f"Val labels - Normal: {(y_val == 0).sum()}, Abnormal: {(y_val == 1).sum()}")
    log.info(f"Test labels - Normal: {(y_test == 0).sum()}, Abnormal: {(y_test == 1).sum()}")
    
    # Train classifier
    log.info("Training XGBoost classifier on combined features...")
    try:
        model, train_metrics = train_classifier(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed,
        )
    except Exception as e:
        log.error(f"Error during XGBoost training: {e}")
        log.error(f"Error type: {type(e)}")
        import traceback
        log.error(traceback.format_exc())
        raise
    
    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = model.predict(X_test)
    
    test_metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "confusion_matrix": confusion_matrix(y_test, test_preds),
        "probs": test_probs,
        "preds": test_preds,
    }
    
    log.info("=" * 60)
    log.info("Test Results:")
    log.info("=" * 60)
    log.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    log.info(f"Precision: {test_metrics['precision']:.4f}")
    log.info(f"Recall:    {test_metrics['recall']:.4f}")
    log.info(f"F1:        {test_metrics['f1']:.4f}")
    log.info(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
    log.info(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save model
    import joblib
    joblib.dump(model, out_dir / "ensemble_xgboost_model.joblib")
    log.info(f"Model saved to {out_dir / 'ensemble_xgboost_model.joblib'}")
    
    # Visualizations
    # 1. ROC Curve
    plt.figure(figsize=(10, 6))
    RocCurveDisplay.from_predictions(y_test, test_metrics["probs"], name="Ensemble XGBoost")
    plt.title("ROC Curve - Ensemble XGBoost", fontsize=20, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ensemble_roc_curve.png", dpi=150)
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_metrics["confusion_matrix"],
        display_labels=["Normal", "Abnormal"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Ensemble XGBoost", fontsize=20, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "ensemble_confusion_matrix.png", dpi=150)
    plt.close()
    
    # 3. Feature Importance
    feature_importance = train_metrics["feature_importance"]
    
    # Group by lead
    features_per_lead = 1 + latent_dim  # MSE + latent
    lead_importance = []
    for i in range(len(LEADS)):
        start_idx = i * features_per_lead
        end_idx = start_idx + features_per_lead
        lead_imp = feature_importance[start_idx:end_idx].sum()
        lead_importance.append(lead_imp)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(LEADS))
    bars = plt.bar(x, lead_importance, color='steelblue', alpha=0.7)
    plt.xlabel("Lead", fontsize=18)
    plt.ylabel("Total Feature Importance", fontsize=18)
    plt.title("Feature Importance by Lead - XGBoost", fontsize=20, fontweight='bold')
    plt.xticks(x, LEADS, fontsize=16)
    plt.tick_params(labelsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, imp in zip(bars, lead_importance):
        plt.text(bar.get_x() + bar.get_width()/2., imp,
               f'{imp:.1f}',
               ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_dir / "ensemble_feature_importance.png", dpi=150)
    plt.close()
    
    # 4. Training curves (loss and AUC)
    if "train_loss" in train_metrics and "val_loss" in train_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        train_loss = train_metrics["train_loss"]
        val_loss = train_metrics["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        
        axes[0].plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
        axes[0].plot(epochs, val_loss, label="Val Loss", color="red", linewidth=2)
        axes[0].set_xlabel("Iteration", fontsize=18)
        axes[0].set_ylabel("Log Loss", fontsize=18)
        axes[0].set_title("Training and Validation Loss - XGBoost", fontsize=18, fontweight='bold')
        axes[0].legend(fontsize=16)
        axes[0].tick_params(labelsize=16)
        axes[0].grid(True, alpha=0.3)
        
        # AUC curves
        if "train_auc" in train_metrics and "val_auc_history" in train_metrics:
            train_auc = train_metrics["train_auc"]
            val_auc_history = train_metrics["val_auc_history"]
            
            axes[1].plot(epochs, train_auc, label="Train AUC", color="green", linewidth=2)
            axes[1].plot(epochs, val_auc_history, label="Val AUC", color="orange", linewidth=2)
            axes[1].axhline(y=train_metrics["val_auc"], color="red", linestyle="--", 
                           label=f"Final Val AUC: {train_metrics['val_auc']:.4f}", alpha=0.7)
            axes[1].set_xlabel("Iteration", fontsize=18)
            axes[1].set_ylabel("AUC", fontsize=18)
            axes[1].set_title("Training and Validation AUC - XGBoost", fontsize=18, fontweight='bold')
            axes[1].legend(fontsize=16)
            axes[1].tick_params(labelsize=16)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close()
    
    # 5. Картинки сырого ЭКГ: 8 отведений на одном изображении, 5 периодов, одна легенда
    #    Только для верно диагностированных пациентов тестовой выборки: 5 больных и 5 здоровых
    if generate_raw_ecg_figures:
        log.info("Generating raw ECG figures (8 leads, 5 periods) for correctly diagnosed patients...")
        try:
            if use_ph_style:
                # Ph: тест — один массив, индексы 0..n_test-1
                correctly_predicted = (y_test == test_preds)
                correct_indices = np.where(correctly_predicted)[0]
                correct_labels = y_test[correct_indices]
                healthy_correct_idx = correct_indices[correct_labels == 0]
                sick_correct_idx = correct_indices[correct_labels == 1]
            else:
                # Per-lead: пересчитываем split тестовой выборки с тем же seed
                test_df_ref, _ = load_lead_data(test_dir, LEADS[0], "test")
                test_full = ECGLeadDataset(test_df_ref)
                val_size = int(len(test_full) * val_fraction)
                test_size = len(test_full) - val_size
                val_subset, test_subset = random_split(
                    test_full,
                    [val_size, test_size],
                    generator=torch.Generator().manual_seed(seed),
                )
                test_indices_in_original = np.array(test_subset.indices)
                correctly_predicted = (y_test == test_preds)
                correct_indices_in_test = np.where(correctly_predicted)[0]
                correct_indices_in_original = test_indices_in_original[correct_indices_in_test]
                correct_labels_in_original = test_df_ref["label"].iloc[correct_indices_in_original].astype(int).to_numpy()
                healthy_correct_idx = correct_indices_in_original[correct_labels_in_original == 0]
                sick_correct_idx = correct_indices_in_original[correct_labels_in_original == 1]

            n_healthy = min(5, len(healthy_correct_idx))
            n_sick = min(5, len(sick_correct_idx))

            if n_healthy == 0 or n_sick == 0:
                log.warning(f"Not enough correctly diagnosed patients: healthy={len(healthy_correct_idx)}, sick={len(sick_correct_idx)}")
                log.info("Skipping raw ECG figure generation")
            else:
                np.random.seed(seed)
                chosen_healthy = np.random.choice(healthy_correct_idx, size=n_healthy, replace=False)
                chosen_sick = np.random.choice(sick_correct_idx, size=n_sick, replace=False)

                log.info(f"Selected {n_healthy} correctly diagnosed healthy and {n_sick} correctly diagnosed sick patients")

                load_signals_fn = load_patient_signals_ph if use_ph_style else load_patient_signals_all_leads
                data_dir_for_signals = dataset_dir if use_ph_style else test_dir

                for k, idx in enumerate(chosen_healthy):
                    if use_ph_style:
                        signals_list, common_len = load_patient_signals_ph(
                            data_dir_for_signals, int(idx), artifacts_base_dir
                        )
                    else:
                        signals_list, common_len = load_patient_signals_all_leads(data_dir_for_signals, int(idx))
                    errors_list = get_patient_errors_all_leads(
                        artifacts_base_dir, signals_list, device, latent_dim, common_len
                    )
                    out_path = out_dir / f"raw_ecg_healthy_{k+1}_patient_{idx}.png"
                    generate_raw_ecg_multilead_figure(
                        signals_per_lead=signals_list,
                        output_path=out_path,
                        patient_label="Здоровый",
                        patient_idx=idx,
                        n_periods=n_periods,
                        errors_per_lead=errors_list,
                    )
                    log.info(f"Saved {out_path}")
                for k, idx in enumerate(chosen_sick):
                    if use_ph_style:
                        signals_list, common_len = load_patient_signals_ph(
                            data_dir_for_signals, int(idx), artifacts_base_dir
                        )
                    else:
                        signals_list, common_len = load_patient_signals_all_leads(data_dir_for_signals, int(idx))
                    errors_list = get_patient_errors_all_leads(
                        artifacts_base_dir, signals_list, device, latent_dim, common_len
                    )
                    out_path = out_dir / f"raw_ecg_sick_{k+1}_patient_{idx}.png"
                    generate_raw_ecg_multilead_figure(
                        signals_per_lead=signals_list,
                        output_path=out_path,
                        patient_label="Больной",
                        patient_idx=idx,
                        n_periods=n_periods,
                        errors_per_lead=errors_list,
                    )
                    log.info(f"Saved {out_path}")
        except Exception as e:
            log.warning(f"Failed to generate raw ECG figures: {e}")
            import traceback
            log.warning(traceback.format_exc())
    
    # Save report
    with open(out_dir / "ensemble_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Ensemble XGBoost Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classifier: XGBoost\n")
        f.write(f"Number of leads: {len(LEADS)}\n")
        f.write(f"Latent dimension per lead: {latent_dim}\n")
        f.write(f"Total features: {X_train.shape[1]} (8 leads × (1 MSE + {latent_dim} latent))\n")
        f.write(f"Training samples (full train2): {X_train.shape[0]}\n")
        f.write(f"Validation samples (from test): {X_val.shape[0]}\n")
        f.write(f"Test samples (from test): {X_test.shape[0]}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Training Metrics:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Validation AUC: {train_metrics['val_auc']:.4f}\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"max_depth: {max_depth}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Test Metrics:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1:        {test_metrics['f1']:.4f}\n")
        f.write(f"ROC AUC:   {test_metrics['roc_auc']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}\n")
        f.write("\n")
        f.write(classification_report(y_test, test_metrics['preds'], digits=4))
        
        # Feature importance by lead
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Feature Importance by Lead:\n")
        f.write("=" * 60 + "\n")
        for lead, imp in zip(LEADS, lead_importance):
            f.write(f"  {lead}: {imp:.4f}\n")
    
    log.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
