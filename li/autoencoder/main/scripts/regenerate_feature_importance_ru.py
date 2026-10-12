#!/usr/bin/env python3
"""Regenerate ensemble feature importance plot with Russian labels (originals unchanged)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

LEADS = ["i", "ii", "v1", "v2", "v3", "v4", "v5", "v6"]

# Values from the reference amy plot used in the thesis description.
AMY_REFERENCE_VALUES = [0.118, 0.150, 0.110, 0.117, 0.130, 0.117, 0.140, 0.115]


def fmt_comma_1(value: float, _pos=None) -> str:
    return f"{value:.1f}".replace(".", ",")


def fmt_comma_2(value: float, _pos=None) -> str:
    return f"{value:.2f}".replace(".", ",")


def fmt_comma_3(value: float, _pos=None) -> str:
    return f"{value:.3f}".replace(".", ",")


def load_lead_importance(
    artifacts_dir: Path,
    latent_dim: int = 16,
) -> list[float]:
    model_path = artifacts_dir / "ensemble_xgboost_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    feature_importance = model.feature_importances_
    features_per_lead = 1 + latent_dim
    lead_importance = []
    for i in range(len(LEADS)):
        start_idx = i * features_per_lead
        end_idx = start_idx + features_per_lead
        lead_importance.append(float(feature_importance[start_idx:end_idx].sum()))
    return lead_importance


def plot_feature_importance(
    lead_importance: list[float],
    output_dir: Path,
    output_name: str = "ensemble_feature_importance.png",
    *,
    compact_y_axis: bool = False,
) -> Path:
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LEADS))
    bars = ax.bar(x, lead_importance, color="steelblue", alpha=0.7)

    ax.set_xlabel("Отведение", fontsize=20)
    ax.set_ylabel("Важность признака", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(LEADS, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)

    if compact_y_axis:
        y_max = max(lead_importance)
        y_top = max(0.15, np.ceil(y_max / 0.02) * 0.02)
        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_comma_2))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_comma_1))

    ax.grid(True, alpha=0.3, axis="y")

    for bar, imp in zip(bars, lead_importance):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            imp,
            fmt_comma_3(imp),
            ha="center",
            va="bottom",
            fontsize=18,
        )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


DEFAULT_ARTIFACTS = [
    (Path("artifacts_ensemble_xgb"), "ensemble_feature_importance.png", False, None),
    (Path("artifacts_ensemble_xgb_ph"), "ensemble_feature_importance_ph.png", False, None),
]

EXTRA_ARTIFACTS = [
    (
        Path("/Users/moomintroll/Desktop/Научка_мага/Autoencoder 2/artifacts_ensemble_xgb_lef"),
        "ensemble_feature_importance_lef.png",
        False,
        None,
    ),
    (
        Path("/Users/moomintroll/Desktop/Научка_мага/Autoencoder 2/artifacts_ensemble_xgb_amy"),
        "ensemble_feature_importance_amy.png",
        True,
        AMY_REFERENCE_VALUES,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Single artifacts directory (optional; default: process all known pathologies)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts_ensemble_xgb_formatted"),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename when using --artifacts_dir",
    )
    parser.add_argument("--latent_dim", type=int, default=16)
    args = parser.parse_args()

    if args.artifacts_dir is not None:
        output_name = args.output_name or "ensemble_feature_importance.png"
        lead_importance = load_lead_importance(args.artifacts_dir, args.latent_dim)
        out_path = plot_feature_importance(
            lead_importance=lead_importance,
            output_dir=args.output_dir,
            output_name=output_name,
        )
        print(f"Saved: {out_path}")
        return

    targets = DEFAULT_ARTIFACTS + EXTRA_ARTIFACTS
    for artifacts_dir, output_name, compact_y_axis, override_values in targets:
        if override_values is None and not (artifacts_dir / "ensemble_xgboost_model.joblib").exists():
            print(f"Skip (model not found): {artifacts_dir}")
            continue

        if override_values is not None:
            lead_importance = override_values
        else:
            lead_importance = load_lead_importance(artifacts_dir, args.latent_dim)

        out_path = plot_feature_importance(
            lead_importance=lead_importance,
            output_dir=args.output_dir,
            output_name=output_name,
            compact_y_axis=compact_y_axis,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
