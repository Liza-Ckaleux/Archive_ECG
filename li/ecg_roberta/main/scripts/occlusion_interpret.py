#!/usr/bin/env python
"""
Occlusion-интерпретация BertBiLSTMClassifier: важность сегментов ЭКГ по отведениям.

Пример:
  python scripts/occlusion_interpret.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_io import (  # noqa: E402
    REFERENCE_LEAD,
    TARGET_CHANNELS,
    build_aligned_beat_windows,
    discover_edf_paths,
    discretize_normalized,
    lead_boundaries_in_normalized,
    load_lef_names,
    map_native_segments_to_normalized,
    normalize_and_discretize,
    process_ecg_data,
    read_edf_path,
    segment_beats_neurokit,
    segment_beats_per_lead,
)

DATASETS_ROOT = Path("/Users/moomintroll/Desktop/Научка_мага/datasets")
MODEL_DIR = REPO_ROOT / "ECGApp" / "ecg_app" / "model"
STATE_DICT_PATH = REPO_ROOT / "ECGApp" / "ecg_app" / "state_dict.pt"
OUTPUT_DIR = REPO_ROOT / "interpretation_outputs"
DECISION_THRESHOLD = 0.5
# 0 = 1-й комплекс, 1 = 2-й, 2 = 3-й (не используем начало записи)
BEAT_INDEX = 2
N_SICK = 4
N_HEALTHY = 4
MAX_CANDIDATES_SCAN = 120

# Окно occlusion в отсчётах (250 Гц): ~128 мс
OCCLUSION_WINDOW = 32
OCCLUSION_STRIDE = 16
MASK_VALUE = 0.5  # после min-max нормализации

SEGMENT_COLORS = {
    "P": "#4C78A8",
    "QRS": "#E45756",
    "ST": "#F58518",
    "T": "#54A24B",
    "baseline": "#BAB0AC",
}


class BertBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model_path, bert_frozen_layers=3, input_size=768, lstm_hidden_size=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_path)
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:bert_frozen_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids)[0]
        lstm_out, _ = self.lstm(bert_output)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BertBiLSTMClassifier(str(MODEL_DIR))
    state = torch.load(STATE_DICT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_proba(model, tokenizer, text, device):
    inputs = tokenizer(
        [text],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logit = model(inputs["input_ids"])
    return torch.sigmoid(logit).item()


def occlusion_importance(normalized, model, tokenizer, device):
    base_text, _ = discretize_normalized(normalized)
    base_p = predict_proba(model, tokenizer, base_text, device)

    n = len(normalized)
    importance = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)

    windows = list(range(0, max(1, n - OCCLUSION_WINDOW + 1), OCCLUSION_STRIDE))
    for wi, start in enumerate(windows):
        end = min(n, start + OCCLUSION_WINDOW)
        masked = normalized.copy()
        masked[start:end] = MASK_VALUE
        text, _ = discretize_normalized(masked)
        p = predict_proba(model, tokenizer, text, device)
        if wi % 10 == 0:
            print(f"    occlusion {wi+1}/{len(windows)}", flush=True)
        drop = base_p - p
        importance[start:end] += np.maximum(drop, 0.0)
        counts[start:end] += 1.0

    counts[counts == 0] = 1.0
    importance /= counts
    if importance.max() > 0:
        importance /= importance.max()
    return importance, base_p


def align_lead_bounds(normalized_len, bounds, resampled_lens):
    total = sum(resampled_lens)
    if total == 0:
        return [0, normalized_len]
    scaled = [0]
    for length in resampled_lens:
        scaled.append(scaled[-1] + int(round(length / total * normalized_len)))
    scaled[-1] = normalized_len
    return scaled


def aggregate_segment_scores(importance, lead_bounds, lead_periods, shared_segments, shared_baseline, sfreq):
    """Суммарная важность по (отведение, тип сегмента); сегменты общие для всех отведений."""
    scores = {ch: {k: 0.0 for k in list(SEGMENT_COLORS)} for ch in TARGET_CHANNELS}
    counts = {ch: {k: 0 for k in SEGMENT_COLORS} for ch in TARGET_CHANNELS}

    for i, ch in enumerate(TARGET_CHANNELS):
        l0, l1 = lead_bounds[i], lead_bounds[i + 1]
        native_len = len(lead_periods[ch])
        seg_n, base_n = map_native_segments_to_normalized(
            shared_segments, shared_baseline, l0, l1, native_len
        )
        for name, (s, e) in seg_n.items():
            s, e = max(l0, s), min(l1, e)
            if e > s:
                scores[ch][name] += importance[s:e].sum()
                counts[ch][name] += e - s
        for s, e in base_n:
            s, e = max(l0, s), min(l1, e)
            if e > s:
                scores[ch]["baseline"] += importance[s:e].sum()
                counts[ch]["baseline"] += e - s

    for ch in TARGET_CHANNELS:
        for k in scores[ch]:
            if counts[ch][k] > 0:
                scores[ch][k] /= counts[ch][k]
    return scores


def map_importance_to_display(imp_model, n_display):
    if len(imp_model) == 0:
        return np.zeros(n_display)
    x_model = np.linspace(0.0, 1.0, len(imp_model))
    x_disp = np.linspace(0.0, 1.0, n_display)
    return np.interp(x_disp, x_model, imp_model)


def plot_lead_importance(
    record_id,
    label,
    prob,
    display_windows,
    r_offset,
    shared_segments,
    shared_baseline,
    lead_bounds,
    importance,
    sfreq,
    out_path,
):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=False)
    fig.suptitle(
        f"{record_id} | {label} | P(LFV)= {prob:.3f} | комплекс #{BEAT_INDEX + 1}\n"
        f"Сегменты: neurokit, якорь {REFERENCE_LEAD}; цвет точек — occlusion",
        fontsize=12,
    )

    for idx, ch in enumerate(TARGET_CHANNELS):
        ax = axes[idx // 2, idx % 2]
        signal = display_windows[ch]
        t = np.arange(len(signal)) / sfreq
        l0, l1 = lead_bounds[idx], lead_bounds[idx + 1]
        imp_slice = map_importance_to_display(importance[l0:l1], len(signal))

        segments, baseline, r_idx = shared_segments, shared_baseline, r_offset
        for seg_name, (s, e) in segments.items():
            e = min(e, len(signal))
            if e > s:
                ax.axvspan(t[s], t[e - 1], color=SEGMENT_COLORS[seg_name], alpha=0.18)
        for s, e in baseline:
            e = min(e, len(signal))
            if e > s:
                ax.axvspan(t[s], t[e - 1], color=SEGMENT_COLORS["baseline"], alpha=0.10)

        ax.plot(t, signal, color="#333333", linewidth=1.0, alpha=0.85, zorder=1)
        ax.scatter(
            t,
            signal,
            c=imp_slice,
            cmap="Reds",
            s=14,
            vmin=0,
            vmax=1,
            zorder=3,
            edgecolors="none",
        )
        ax.scatter([t[r_idx]], [signal[r_idx]], color="lime", s=45, zorder=4, marker="x", linewidths=1.5)
        pad = 0.05 * max(np.ptp(signal), 1e-6)
        ax.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 0.1)
        ax.set_ylim(signal.min() - pad, signal.max() + pad)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("с")
        ax.set_ylabel("амплитуда")
        ax.grid(True, alpha=0.25)

    legend = [Patch(facecolor=c, alpha=0.35, label=n) for n, c in SEGMENT_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_heatmap(record_id, label, scores, out_path):
    seg_names = ["baseline", "P", "QRS", "ST", "T"]
    mat = np.array([[scores[ch].get(s, 0.0) for s in seg_names] for ch in TARGET_CHANNELS])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(mat.max(), 1e-6))
    ax.set_xticks(range(len(seg_names)))
    ax.set_xticklabels(seg_names)
    ax.set_yticks(range(len(TARGET_CHANNELS)))
    ax.set_yticklabels(TARGET_CHANNELS)
    ax.set_title(f"Средняя важность по сегментам — {record_id} ({label})")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="норм. важность")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_segments(record_id, label, scores, out_path):
    totals = {k: 0.0 for k in SEGMENT_COLORS}
    for ch in TARGET_CHANNELS:
        for k, v in scores[ch].items():
            totals[k] += v
    names = list(SEGMENT_COLORS.keys())
    vals = [totals[k] for k in names]
    colors = [SEGMENT_COLORS[k] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("суммарная норм. важность")
    ax.set_title(f"Вклад типов сегментов — {record_id} ({label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def predict_record_prob(edf_path, model, tokenizer, device, beat_index=BEAT_INDEX):
    signals, channel_labels, sfreq = read_edf_path(edf_path)
    combined, _ = process_ecg_data(signals, channel_labels, sfreq, beat_index=beat_index)
    _, text, _ = normalize_and_discretize(combined, sfreq)
    return predict_proba(model, tokenizer, text, device)


def is_correctly_classified(label, prob):
    expected_positive = "больной" in label
    if expected_positive:
        return prob >= DECISION_THRESHOLD
    return prob < DECISION_THRESHOLD


def build_display_windows(signals, channel_labels, sfreq, beat_index=BEAT_INDEX):
    windows, r_offset, win_start, win_end = build_aligned_beat_windows(
        signals, channel_labels, sfreq, beat_index=beat_index
    )
    try:
        segments, baseline, r_offset = segment_beats_neurokit(
            signals,
            channel_labels,
            sfreq,
            beat_index=beat_index,
            window_start=win_start,
            window_end=win_end,
            r_offset=r_offset,
        )
    except Exception as exc:
        print(f"  neurokit delineate fallback: {exc}")
        ref_signal = windows[REFERENCE_LEAD]
        segments, baseline, r_offset = segment_beats_per_lead(
            ref_signal, sfreq, r_idx=r_offset
        )
    return windows, r_offset, segments, baseline


def select_correct_records(paths, lef, model, tokenizer, device):
    import random

    sick_pool = [n for n in lef if n in paths]
    healthy_pool = [n for n in paths if n not in lef]
    random.shuffle(sick_pool)
    random.shuffle(healthy_pool)

    selected = []

    def fill(pool, label, need_positive, target_count):
        found = 0
        for name in pool[:MAX_CANDIDATES_SCAN]:
            try:
                prob = predict_record_prob(paths[name], model, tokenizer, device)
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

    fill(sick_pool, "больной (LFV)", need_positive=True, target_count=N_SICK)
    fill(healthy_pool, "здоровый (noLFV)", need_positive=False, target_count=N_HEALTHY)
    return selected


def process_record(
    edf_path, record_id, label, model, tokenizer, device, prob_hint=None, beat_index=BEAT_INDEX
):
    signals, channel_labels, sfreq = read_edf_path(edf_path)
    combined, lead_periods = process_ecg_data(
        signals, channel_labels, sfreq, beat_index=beat_index
    )
    display_windows, r_offset, shared_segments, shared_baseline = build_display_windows(
        signals, channel_labels, sfreq, beat_index=beat_index
    )
    normalized, text, _ = normalize_and_discretize(combined, sfreq)
    bounds, lens = lead_boundaries_in_normalized(lead_periods, sfreq=sfreq)
    bounds = align_lead_bounds(len(normalized), bounds, lens)

    print("  occlusion...")
    importance, prob = occlusion_importance(normalized, model, tokenizer, device)
    if prob_hint is not None:
        prob = prob_hint
    scores = aggregate_segment_scores(
        importance, bounds, lead_periods, shared_segments, shared_baseline, sfreq
    )

    stem = f"{record_id}_{label.replace(' ', '_')}"
    plot_lead_importance(
        record_id,
        label,
        prob,
        display_windows,
        r_offset,
        shared_segments,
        shared_baseline,
        bounds,
        importance,
        sfreq,
        OUTPUT_DIR / f"{stem}_leads.png",
    )
    plot_segment_heatmap(record_id, label, scores, OUTPUT_DIR / f"{stem}_heatmap.png")
    plot_global_segments(record_id, label, scores, OUTPUT_DIR / f"{stem}_segments.png")

    return {"prob": prob, "scores": scores}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print("Загрузка модели...")
    model, tokenizer = load_model(device)

    lef = load_lef_names(DATASETS_ROOT / "lef_names.csv")
    paths = discover_edf_paths(DATASETS_ROOT)

    print(f"Комплекс для анализа: #{BEAT_INDEX + 1} (beat_index={BEAT_INDEX})")
    print("Поиск верно классифицированных записей...")
    selected = select_correct_records(paths, lef, model, tokenizer, device)
    if len(selected) < N_SICK + N_HEALTHY:
        print(f"Предупреждение: найдено только {len(selected)} из {N_SICK + N_HEALTHY} записей")
    print("Выбрано:", [(n, lab, f"{p:.3f}") for n, _, lab, p in selected])

    summary = []
    for record_id, edf_path, label, prob_hint in selected:
        print(f"\n=== {record_id} ({label}) ===")
        try:
            res = process_record(
                edf_path, record_id, label, model, tokenizer, device, prob_hint=prob_hint
            )
            summary.append((record_id, label, res["prob"], "ok"))
            print(f"  P(LFV)={res['prob']:.3f}")
            print(f"  Сохранено в {OUTPUT_DIR}")
        except Exception as exc:
            print(f"  Ошибка: {exc}")
            summary.append((record_id, label, prob_hint, f"error: {exc}"))

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# beat_index={BEAT_INDEX} (комплекс #{BEAT_INDEX + 1})\n")
        f.write("record_id\tlabel\tP(LFV)\tstatus\n")
        for rid, lab, p, st in summary:
            f.write(f"{rid}\t{lab}\t{p:.4f}\t{st}\n")
    print("\nГотово:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
