#!/usr/bin/env python3
"""
Train a per-lead autoencoder on normal ECG beats only.

Pipeline:
1. Train AE on label == 0
2. Save ae_model.pt and basic diagnostics on val/test split
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.autoencoder import AutoencoderConfig, ECGAutoencoder


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


def split_val_test(dataset: Dataset, val_fraction: float, seed: int):
    n = len(dataset)
    val_n = int(n * val_fraction)
    test_n = n - val_n
    return random_split(
        dataset,
        [val_n, test_n],
        generator=torch.Generator().manual_seed(seed),
    )


def train_autoencoder(
    model: ECGAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rec_loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total = 0.0
        n = 0

        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = rec_loss_fn(out["reconstruction"], x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

        losses.append(total / max(1, n))

    return losses


def collect_reconstruction_errors(
    model: ECGAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mse = nn.MSELoss(reduction="none")
    errors: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            err = mse(out["reconstruction"], x).mean(dim=1).cpu().numpy()
            errors.append(err)
            labels.append(y.cpu().numpy())

    return np.concatenate(errors), np.concatenate(labels).astype(int)


def main() -> None:
    with open("configs/training_params.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Train per-lead ECG autoencoder")
    parser.add_argument("--train", type=Path, default=Path(cfg.get("train")))
    parser.add_argument("--test", type=Path, default=Path(cfg.get("test")))
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 60))
    parser.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 32))
    parser.add_argument("--latent_dim", type=int, default=cfg.get("latent_dim", 16))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 5e-4))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--val_fraction", type=float, default=cfg.get("val_fraction", 1 / 3))
    parser.add_argument("--out", type=Path, default=Path(cfg.get("output", "artifacts_temp")))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("train-autoencoder")

    train_df = pd.read_pickle(args.train)
    test_df = pd.read_pickle(args.test)
    train_df_norm = train_df[train_df.label == 0].reset_index(drop=True)

    train_set = ECGLeadDataset(train_df_norm)
    test_full = ECGLeadDataset(test_df, target_length=train_set.target_length)
    val_set, test_set = split_val_test(test_full, args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_x, _ = train_set[0]
    model = ECGAutoencoder(
        AutoencoderConfig(
            input_dim=sample_x.numel(),
            latent_dim=args.latent_dim,
        )
    ).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    log.info("Training autoencoder on NORMAL data only...")
    train_losses = train_autoencoder(model, train_loader, device, args.epochs, args.lr)
    torch.save(model.state_dict(), args.out / "ae_model.pt")

    val_err, y_val = collect_reconstruction_errors(model, val_loader, device)
    test_err, y_test = collect_reconstruction_errors(model, test_loader, device)

    auc_val = roc_auc_score(y_val, val_err) if len(np.unique(y_val)) > 1 else float("nan")
    auc_test = roc_auc_score(y_test, test_err) if len(np.unique(y_test)) > 1 else float("nan")
    log.info(f"[val]  AUC (MSE reconstruction error) = {auc_val:.4f}")
    log.info(f"[test] AUC (MSE reconstruction error) = {auc_test:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title("AE training loss (normals)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out / "ae_train_loss.png", dpi=150)
    plt.close()

    if len(np.unique(y_test)) > 1:
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test, test_err, name=f"MSE (AUC={auc_test:.3f})")
        plt.title("ROC curve — reconstruction error")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out / "roc_curves.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(test_err[y_test == 0], bins=50, alpha=0.6, label="Normal", density=True)
        plt.hist(test_err[y_test == 1], bins=50, alpha=0.6, label="Abnormal", density=True)
        plt.xlabel("Reconstruction error (MSE)")
        plt.ylabel("Density")
        plt.title("Reconstruction error distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out / "reconstruction_error_histogram.png", dpi=150)
        plt.close()

    with open(args.out / "ae_report.txt", "w", encoding="utf-8") as f:
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"latent_dim: {args.latent_dim}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"train_samples_normal: {len(train_set)}\n")
        f.write(f"val_auc_mse: {auc_val:.4f}\n")
        f.write(f"test_auc_mse: {auc_test:.4f}\n")

    log.info("Saved model and reports to %s", args.out)


if __name__ == "__main__":
    main()
