#!/usr/bin/env python
"""
Агрегированная occlusion-интерпретация по группе записей.

Пример:
  /Users/moomintroll/Desktop/Научка_мага/ecg_roberta/ECGApp/venv/bin/python \
      scripts/occlusion_global_summary.py --n-sick 6 --n-healthy 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_io import (  # noqa: E402
    TARGET_CHANNELS,
    discover_edf_paths,
    lead_boundaries_in_normalized,
    load_lef_names,
    normalize_and_discretize,
    process_ecg_data,
    read_edf_path,
)
from occlusion_interpret import (  # noqa: E402
    BEAT_INDEX,
    DATASETS_ROOT,
    MAX_CANDIDATES_SCAN,
    SEGMENT_COLORS,
    aggregate_segment_scores,
    align_lead_bounds,
    build_display_windows,
    is_correctly_classified,
    load_model,
    occlusion_importance,
    predict_record_prob,
)

SEGMENT_NAMES = ["baseline", "P", "QRS", "ST", "T"]
OUTPUT_DIR = REPO_ROOT / "interpretation_outputs" / "global_summary"
COLORBAR_LABEL = "Средняя нормированная важность"
AGGREGATE_NPZ_NAME = "aggregate_matrices.npz"


def fmt_decimal(value, places=2):
    return f"{value:.{places}f}".replace(".", ",")


def axis_decimal_formatter(x, _pos):
    return fmt_decimal(x, places=2 if abs(x) < 10 else 1)


def configure_style():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    serif_stack = ["Times New Roman", "Times", "DejaVu Serif"]
    if "Times New Roman" not in available_fonts:
        serif_stack = ["Times", "DejaVu Serif"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_stack,
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
        }
    )


def select_correct_records(
    paths,
    lef_names,
    model,
    tokenizer,
    device,
    n_sick,
    n_healthy,
    beat_index,
    max_candidates_scan,
    seed,
):
    rng = random.Random(seed)
    sick_pool = [name for name in lef_names if name in paths]
    healthy_pool = [name for name in paths if name not in lef_names]
    rng.shuffle(sick_pool)
    rng.shuffle(healthy_pool)

    selected = []

    def fill(pool, label, target_count):
        found = 0
        for name in pool[:max_candidates_scan]:
            try:
                prob = predict_record_prob(paths[name], model, tokenizer, device, beat_index=beat_index)
            except Exception as exc:
                print(f"  skip {name}: {exc}")
                continue
            ok = is_correctly_classified(label, prob)
            print(f"  probe {name}: P(LFV)={prob:.3f} -> {'OK' if ok else 'skip'}")
            if ok:
                selected.append((name, paths[name], label, prob))
                found += 1
            if found >= target_count:
                break

    fill(sick_pool, "больной (LFV)", n_sick)
    fill(healthy_pool, "здоровый (noLFV)", n_healthy)
    return selected


def scores_to_matrix(scores):
    return np.array(
        [[scores[ch].get(segment, 0.0) for segment in SEGMENT_NAMES] for ch in TARGET_CHANNELS],
        dtype=float,
    )


def normalize_matrix(matrix):
    total = float(matrix.sum())
    if total <= 0:
        return matrix.copy()
    return matrix / total


def compute_record_matrix(edf_path, model, tokenizer, device, beat_index):
    signals, channel_labels, sfreq = read_edf_path(edf_path)
    combined, lead_periods = process_ecg_data(signals, channel_labels, sfreq, beat_index=beat_index)
    _, _, shared_segments, shared_baseline = build_display_windows(
        signals, channel_labels, sfreq, beat_index=beat_index
    )
    normalized, _, _ = normalize_and_discretize(combined, sfreq)
    bounds, lens = lead_boundaries_in_normalized(lead_periods, sfreq=sfreq)
    bounds = align_lead_bounds(len(normalized), bounds, lens)

    print("  occlusion...")
    importance, prob = occlusion_importance(normalized, model, tokenizer, device)
    scores = aggregate_segment_scores(
        importance,
        bounds,
        lead_periods,
        shared_segments,
        shared_baseline,
        sfreq,
    )
    matrix = scores_to_matrix(scores)
    return matrix, normalize_matrix(matrix), float(prob)


def top_k_frequency(matrices, k=3):
    counts = np.zeros_like(matrices[0], dtype=float)
    for mat in matrices:
        flat = mat.ravel()
        top_k = min(k, flat.size)
        idx = np.argpartition(flat, -top_k)[-top_k:]
        counts.ravel()[idx] += 1.0
    return counts / len(matrices)


def plot_heatmap(matrix, out_path, title=None, cmap="YlOrRd", centered=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    if centered:
        vmax = float(np.max(np.abs(matrix))) or 1e-6
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        cbar_vmax = vmax
        cbar_vmin = -vmax
    else:
        cbar_vmax = max(float(matrix.max()), 1e-6)
        cbar_vmin = 0.0
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=cbar_vmin, vmax=cbar_vmax)

    ax.set_xticks(range(len(SEGMENT_NAMES)))
    ax.set_xticklabels(SEGMENT_NAMES)
    ax.set_yticks(range(len(TARGET_CHANNELS)))
    ax.set_yticklabels(TARGET_CHANNELS)
    if title:
        ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                fmt_decimal(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, label=COLORBAR_LABEL)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(axis_decimal_formatter))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_segment_bars(mean_lfv, mean_nolfv, out_path):
    lfv_vals = mean_lfv.sum(axis=0)
    nolfv_vals = mean_nolfv.sum(axis=0)
    x = np.arange(len(SEGMENT_NAMES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, lfv_vals, width=width, color="#D55E00", label="класс 1")
    ax.bar(x + width / 2, nolfv_vals, width=width, color="#0072B2", label="класс 0")
    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENT_NAMES)
    ax.set_ylabel("Средняя суммарная важность")
    ax.yaxis.set_major_formatter(FuncFormatter(axis_decimal_formatter))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _rgb_to_scalar(rgb, cmap, vmin, vmax, n_steps=1200):
    if vmax <= vmin:
        return vmin
    best_v, best_err = vmin, 1e9
    for v in np.linspace(vmin, vmax, n_steps):
        c = np.array(cmap((v - vmin) / (vmax - vmin)))[:3]
        err = float(np.sum((c - rgb) ** 2))
        if err < best_err:
            best_err, best_v = err, v
    return best_v


def recover_heatmap_matrix_from_png(
    png_path,
    cmap_name="YlOrRd",
    n_rows=None,
    n_cols=None,
    centered=False,
):
    """Восстановление матрицы из сохранённого heatmap (для --replot-only)."""
    n_rows = n_rows or len(TARGET_CHANNELS)
    n_cols = n_cols or len(SEGMENT_NAMES)
    img = np.asarray(Image.open(png_path).convert("RGB"), dtype=float) / 255.0
    h, w = img.shape[:2]
    x0, x1 = int(0.10 * w), int(0.70 * w)
    y0, y1 = int(0.12 * h), int(0.86 * h)
    cmap = plt.colormaps[cmap_name]
    cbar_x = int(0.79 * w)

    if centered:
        top_rgb = img[int(0.18 * h), cbar_x, :3]
        bot_rgb = img[int(0.80 * h), cbar_x, :3]
        vmax = _rgb_to_scalar(top_rgb, cmap, 0.0, 1.0)
        vmin = -vmax
    else:
        top_rgb = img[int(0.18 * h), cbar_x, :3]
        bot_rgb = img[int(0.80 * h), cbar_x, :3]
        vmax = _rgb_to_scalar(top_rgb, cmap, 0.0, 1.0)
        vmin = _rgb_to_scalar(bot_rgb, cmap, 0.0, vmax)

    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_rows):
        for j in range(n_cols):
            cy = int(y0 + (i + 0.5) * (y1 - y0) / n_rows)
            cx = int(x0 + (j + 0.5) * (x1 - x0) / n_cols)
            matrix[i, j] = _rgb_to_scalar(img[cy, cx, :3], cmap, vmin, vmax)
    return matrix


def save_aggregate_npz(outdir, mean_lfv, mean_nolfv, delta, freq_lfv, freq_nolfv):
    np.savez(
        outdir / AGGREGATE_NPZ_NAME,
        mean_lfv=mean_lfv,
        mean_nolfv=mean_nolfv,
        delta=delta,
        freq_lfv=freq_lfv,
        freq_nolfv=freq_nolfv,
    )


def load_aggregate_npz(outdir):
    path = outdir / AGGREGATE_NPZ_NAME
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def load_selection_from_summary(outdir):
    summary_path = Path(outdir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Не найден {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    return summary["selected"]["lfv"], summary["selected"]["nolfv"], summary.get("beat_index", BEAT_INDEX)


def build_selected_from_summary(paths, lef_names, lfv_ids, nolfv_ids):
    selected = []
    for record_id in lfv_ids:
        if record_id not in paths:
            raise KeyError(f"Запись {record_id} не найдена среди EDF")
        selected.append((record_id, paths[record_id], "больной (LFV)", None))
    for record_id in nolfv_ids:
        if record_id not in paths:
            raise KeyError(f"Запись {record_id} не найдена среди EDF")
        selected.append((record_id, paths[record_id], "здоровый (noLFV)", None))
    return selected


def replot_aggregates(outdir, data=None, allow_png_recovery=False):
    outdir = Path(outdir)
    if data is None:
        data = load_aggregate_npz(outdir)
    freq_lfv = None
    freq_nolfv = None
    if data is not None:
        mean_lfv = data["mean_lfv"]
        mean_nolfv = data["mean_nolfv"]
        delta = data["delta"]
        freq_lfv = data.get("freq_lfv")
        freq_nolfv = data.get("freq_nolfv")
    elif allow_png_recovery:
        lfv_path = outdir / "mean_heatmap_lfv.png"
        nolfv_path = outdir / "mean_heatmap_nolfv.png"
        if not lfv_path.exists() or not nolfv_path.exists():
            raise FileNotFoundError(
                f"Нет {AGGREGATE_NPZ_NAME} и не найдены heatmap PNG в {outdir}"
            )
        print(
            "ВНИМАНИЕ: матрицы восстанавливаются из PNG (приближённо, значения искажаются). "
            "Для точных данных используйте --from-summary без --replot-only."
        )
        mean_lfv = recover_heatmap_matrix_from_png(lfv_path)
        mean_nolfv = recover_heatmap_matrix_from_png(nolfv_path)
        delta = mean_lfv - mean_nolfv
    else:
        raise FileNotFoundError(
            f"Нет {outdir / AGGREGATE_NPZ_NAME}. "
            "Перестройка только оформления возможна после сохранения точных матриц "
            "(полный прогон или --from-summary). "
            "Восстановление из PNG: --replot-only --recover-from-png (неточно)."
        )

    plot_heatmap(mean_lfv, outdir / "mean_heatmap_lfv.png")
    plot_heatmap(mean_nolfv, outdir / "mean_heatmap_nolfv.png")
    plot_heatmap(
        delta,
        outdir / "delta_heatmap_lfv_minus_nolfv.png",
        cmap="coolwarm",
        centered=True,
    )
    plot_segment_bars(mean_lfv, mean_nolfv, outdir / "global_segments_barplot.png")
    if freq_lfv is not None and freq_nolfv is not None:
        plot_top3_stability(freq_lfv, freq_nolfv, outdir / "top3_stability.png")
    print("Перестроены графики в", outdir)


def plot_top3_stability(freq_lfv, freq_nolfv, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, matrix, title in zip(
        axes,
        [freq_lfv, freq_nolfv],
        ["LFV: доля попаданий в top-3", "noLFV: доля попаданий в top-3"],
    ):
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(SEGMENT_NAMES)))
        ax.set_xticklabels(SEGMENT_NAMES)
        ax.set_yticks(range(len(TARGET_CHANNELS)))
        ax.set_yticklabels(TARGET_CHANNELS)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt_decimal(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label="частота")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(axis_decimal_formatter))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def top_cells(matrix, n=5):
    flat = matrix.ravel()
    idx = np.argsort(flat)[::-1][:n]
    result = []
    for index in idx:
        lead_idx, seg_idx = np.unravel_index(index, matrix.shape)
        result.append(
            {
                "lead": TARGET_CHANNELS[lead_idx],
                "segment": SEGMENT_NAMES[seg_idx],
                "value": float(matrix[lead_idx, seg_idx]),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Агрегированный анализ occlusion-важности.")
    parser.add_argument("--n-sick", type=int, default=6, help="Сколько верно классифицированных LFV взять")
    parser.add_argument(
        "--n-healthy",
        type=int,
        default=6,
        help="Сколько верно классифицированных noLFV взять",
    )
    parser.add_argument("--beat-index", type=int, default=BEAT_INDEX, help="Индекс комплекса")
    parser.add_argument(
        "--max-candidates-scan",
        type=int,
        default=MAX_CANDIDATES_SCAN,
        help="Сколько кандидатов сканировать в каждом классе",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимого выбора записей")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR), help="Куда сохранить итоговые графики")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Только перестроить графики из aggregate_matrices.npz (точные данные)",
    )
    parser.add_argument(
        "--recover-from-png",
        action="store_true",
        help="При --replot-only: восстановить матрицы из PNG (неточно, не рекомендуется)",
    )
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Взять список записей из summary.json в --outdir (без нового сканирования)",
    )
    return parser.parse_args()


def run_aggregation(results, outdir, beat_index):
    if not results["больной (LFV)"] or not results["здоровый (noLFV)"]:
        raise RuntimeError("Недостаточно записей для обеих групп")

    lfv_stack = np.stack([item["norm_matrix"] for item in results["больной (LFV)"]])
    nolfv_stack = np.stack([item["norm_matrix"] for item in results["здоровый (noLFV)"]])

    mean_lfv = lfv_stack.mean(axis=0)
    mean_nolfv = nolfv_stack.mean(axis=0)
    delta = mean_lfv - mean_nolfv
    freq_lfv = top_k_frequency(lfv_stack, k=3)
    freq_nolfv = top_k_frequency(nolfv_stack, k=3)

    save_aggregate_npz(outdir, mean_lfv, mean_nolfv, delta, freq_lfv, freq_nolfv)

    plot_heatmap(mean_lfv, outdir / "mean_heatmap_lfv.png")
    plot_heatmap(mean_nolfv, outdir / "mean_heatmap_nolfv.png")
    plot_heatmap(
        delta,
        outdir / "delta_heatmap_lfv_minus_nolfv.png",
        cmap="coolwarm",
        centered=True,
    )
    plot_segment_bars(mean_lfv, mean_nolfv, outdir / "global_segments_barplot.png")
    plot_top3_stability(freq_lfv, freq_nolfv, outdir / "top3_stability.png")

    summary = {
        "beat_index": beat_index,
        "selected": {
            "lfv": [item["record_id"] for item in results["больной (LFV)"]],
            "nolfv": [item["record_id"] for item in results["здоровый (noLFV)"]],
        },
        "top_cells": {
            "lfv": top_cells(mean_lfv),
            "nolfv": top_cells(mean_nolfv),
            "delta_lfv_minus_nolfv": top_cells(delta),
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nГотово:", outdir)


def main():
    configure_style()
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        replot_aggregates(outdir, allow_png_recovery=args.recover_from_png)
        return

    device = torch.device("cpu")
    print("Загрузка модели...")
    model, tokenizer = load_model(device)
    lef_names = load_lef_names(DATASETS_ROOT / "lef_names.csv")
    paths = discover_edf_paths(DATASETS_ROOT)

    beat_index = args.beat_index
    if args.from_summary:
        lfv_ids, nolfv_ids, beat_index = load_selection_from_summary(outdir)
        print(
            f"Список из summary.json: LFV={len(lfv_ids)}, noLFV={len(nolfv_ids)}, "
            f"beat_index={beat_index}"
        )
        selected = build_selected_from_summary(paths, lef_names, lfv_ids, nolfv_ids)
    else:
        print("Поиск верно классифицированных записей...")
        selected = select_correct_records(
            paths,
            lef_names,
            model,
            tokenizer,
            device,
            n_sick=args.n_sick,
            n_healthy=args.n_healthy,
            beat_index=beat_index,
            max_candidates_scan=args.max_candidates_scan,
            seed=args.seed,
        )

    results = {"больной (LFV)": [], "здоровый (noLFV)": []}

    for record_id, edf_path, label, prob_hint in selected:
        print(f"\n=== {record_id} ({label}) ===")
        try:
            raw_matrix, norm_matrix, prob = compute_record_matrix(
                edf_path,
                model,
                tokenizer,
                device,
                beat_index=beat_index,
            )
            results[label].append(
                {
                    "record_id": record_id,
                    "prob": prob_hint if prob_hint is not None else prob,
                    "raw_matrix": raw_matrix,
                    "norm_matrix": norm_matrix,
                }
            )
        except Exception as exc:
            print(f"  skip {record_id}: {exc}")

    run_aggregation(results, outdir, beat_index)


if __name__ == "__main__":
    main()
