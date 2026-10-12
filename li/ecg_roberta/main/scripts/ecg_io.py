"""Чтение EDF и предобработка ЭКГ (без Django)."""
import math
import os
import tempfile

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks

TARGET_CHANNELS = [
    "ECG I", "ECG II", "ECG V1", "ECG V2",
    "ECG V3", "ECG V4", "ECG V5", "ECG V6",
]
REFERENCE_LEAD = "ECG II"
TARGET_SFREQ = 250
DISCRETIZATION_FACTOR = 100
MAX_WINDOW_SIZE = 4000


class DiffQuantizer:
    def __init__(self, alphabet_size, average_over=1, filter=False, breakpoints=None, use_diffs=True):
        self.alphabet_size = alphabet_size
        self.average_over = average_over
        self.use_filter = filter
        self.breakpoints = breakpoints
        self.use_diffs = use_diffs

    def preprocess(self, tmp):
        if self.average_over != 1:
            tmp = self._average_over_n(tmp, self.average_over)
        if self.use_filter:
            tmp = self._filter_signal(tmp)
        if self.use_diffs:
            tmp = self._diff_signal(tmp)
        return tmp

    def perform_quantization(self, tmp, breakpoints=None):
        self.breakpoints = breakpoints
        tmp = self.preprocess(tmp)
        return self._quantize_with_breakpoints(tmp)

    def learn_breakpoints(self, arr):
        res = self.preprocess(arr)
        sorted_array = np.sort(res)
        length = len(sorted_array)
        probs = [1 / self.alphabet_size for _ in range(self.alphabet_size)]
        cum_sum_breakpoints = [int(sum(probs[0 : i + 1]) * length - 1) for i in range(len(probs))]
        cum_sum_breakpoint_values = sorted_array[cum_sum_breakpoints]
        cum_sum_breakpoint_values[-1] = 1e100
        self.breakpoints = cum_sum_breakpoint_values
        return cum_sum_breakpoint_values

    @staticmethod
    def _breakpoint_to_letter(float_num, breakpoints):
        int_val = next((breakpoints.index(obj) for obj in breakpoints if float_num < obj))
        return chr(65 + int_val)

    def _quantize_with_breakpoints(self, tmp):
        breakpoints = self.breakpoints
        vect_breakpoint_to_letter = np.vectorize(self._breakpoint_to_letter, excluded=["breakpoints"])
        return vect_breakpoint_to_letter(tmp, breakpoints=list(breakpoints))

    @staticmethod
    def _average_over_n(tmp, n):
        return np.array([np.average(tmp[i : i + n]) for i in range(0, len(tmp), n)])

    @staticmethod
    def _diff_signal(tmp):
        res = np.diff(tmp)
        return np.insert(res, 0, 0.0)


class Preprocessing:
    def __init__(self, input_file, discretizition_factor, max_window_size):
        self.input_file = input_file
        self.discretizition_factor = discretizition_factor
        self.max_window_size = max_window_size

    def create_window(self):
        ecg_window = []
        ecg_list = self.input_file["signal"]
        mlen = len(ecg_list)
        window_size = min(mlen, self.max_window_size)
        num_lines = math.floor(len(ecg_list) / window_size)
        for i in range(num_lines):
            ecg_window.append(ecg_list[i * window_size : (i + 1) * window_size])
        return ecg_window

    def change_to_alphabet(self, quantizer, normalized_list):
        qtz_signal = []
        for i in range(len(normalized_list)):
            r = quantizer.perform_quantization(np.array(normalized_list[i]), breakpoints=quantizer.breakpoints)
            qtz_signal.append("".join(r))
        return qtz_signal


def resample_signal(signal, original_sampling_rate=500, target_sampling_rate=TARGET_SFREQ):
    return nk.signal_resample(
        signal,
        sampling_rate=original_sampling_rate,
        desired_sampling_rate=target_sampling_rate,
        method="FFT",
    )


def discretization_lloyd_max(discretizition_factor, total_data):
    qtz = DiffQuantizer(alphabet_size=discretizition_factor, breakpoints=None, use_diffs=False)
    qtz.learn_breakpoints(np.array(total_data))
    return qtz


def read_edf_path(file_path):
    f = pyedflib.EdfReader(file_path)
    try:
        n_channels = f.signals_in_file
        channel_labels = f.getSignalLabels()
        signals = np.vstack([f.readSignal(i) for i in range(n_channels)])
        sfreq = f.getSampleFrequency(0)
        return signals, channel_labels, sfreq
    finally:
        f.close()


def find_r_peaks(signal, sfreq, distance_sec=0.6):
    peaks, _ = find_peaks(signal, distance=int(distance_sec * sfreq * 0.8))
    return peaks


def extract_period_at_beat(signal, sfreq, beat_index=2, half_width_sec=0.55, distance_sec=0.6):
    """
    Фрагмент вокруг R-пика (симметричное окно).
    beat_index: 0 — 1-й комплекс, 1 — 2-й, 2 — 3-й и т.д.
    """
    peaks = find_r_peaks(signal, sfreq, distance_sec)
    if len(peaks) == 0:
        raise ValueError("R-пики не найдены")
    if beat_index >= len(peaks):
        beat_index = len(peaks) - 1
    r_idx = int(peaks[beat_index])
    half = int(half_width_sec * sfreq)
    start = max(0, r_idx - half)
    end = min(len(signal), r_idx + half)
    return signal[start:end], r_idx - start, peaks


def extract_periods(signal, sfreq, num_periods=1, distance_sec=0.6):
    """Устаревший вариант (между пиками); оставлен для совместимости."""
    peaks = find_r_peaks(signal, sfreq, distance_sec)
    if len(peaks) < num_periods + 1:
        raise ValueError("Недостаточно пиков")
    periods = []
    for i in range(num_periods):
        start = peaks[i]
        end = peaks[i + 1]
        half = int((end - start) / 2)
        periods.append(signal[end - half : end + half])
    return periods, peaks


def extract_beat_window(
    filtered_signal,
    sfreq,
    beat_index=2,
    pre_sec=0.35,
    post_sec=0.55,
    distance_sec=0.6,
):
    """Окно вокруг R-пика выбранного комплекса для отображения ЭКГ."""
    peaks = find_r_peaks(filtered_signal, sfreq, distance_sec)
    if len(peaks) == 0:
        raise ValueError("R-пики не найдены")
    if beat_index >= len(peaks):
        beat_index = len(peaks) - 1
    r_idx = int(peaks[beat_index])
    start = max(0, r_idx - int(pre_sec * sfreq))
    end = min(len(filtered_signal), r_idx + int(post_sec * sfreq))
    return filtered_signal[start:end], r_idx - start


def filter_channel(signals, channel_labels, sfreq, channel_name):
    lowcut, highcut, order = 0.6, 40, 1
    nyquist_freq = 0.5 * sfreq
    b, a = butter(order, [lowcut / nyquist_freq, highcut / nyquist_freq], btype="bandpass")
    idx = channel_labels.index(channel_name)
    return filtfilt(b, a, signals[idx])


def neurokit_r_peaks(filtered_signal, sfreq):
    clean = nk.ecg_clean(filtered_signal, sampling_rate=sfreq)
    _, peaks = nk.ecg_peaks(clean, sampling_rate=sfreq)
    return clean, [int(x) for x in peaks["ECG_R_Peaks"] if not np.isnan(x)]


def reference_r_absolute(
    signals,
    channel_labels,
    sfreq,
    beat_index=2,
    ref_channel=REFERENCE_LEAD,
):
    ref = filter_channel(signals, channel_labels, sfreq, ref_channel)
    _, rpeaks = neurokit_r_peaks(ref, sfreq)
    if not rpeaks:
        raise ValueError(f"R-пики не найдены на {ref_channel}")
    idx = min(beat_index, len(rpeaks) - 1)
    return int(rpeaks[idx]), ref


def aligned_beat_window_bounds(
    signals,
    channel_labels,
    sfreq,
    beat_index=2,
    pre_sec=0.35,
    post_sec=0.55,
    ref_channel=REFERENCE_LEAD,
):
    """Границы окна по R-пику опорного отведения (одинаковые для всех каналов)."""
    r_abs, ref = reference_r_absolute(
        signals, channel_labels, sfreq, beat_index=beat_index, ref_channel=ref_channel
    )
    start = max(0, r_abs - int(pre_sec * sfreq))
    end = min(len(ref), r_abs + int(post_sec * sfreq))
    return start, end, r_abs - start, r_abs


def slice_aligned_leads(
    signals,
    channel_labels,
    sfreq,
    start,
    end,
    target_channels=None,
):
    target_channels = target_channels or TARGET_CHANNELS
    windows = {}
    for ch in target_channels:
        filtered = filter_channel(signals, channel_labels, sfreq, ch)
        windows[ch] = np.asarray(filtered[start:end], dtype=float)
    return windows


def build_aligned_beat_windows(
    signals,
    channel_labels,
    sfreq,
    beat_index=2,
    pre_sec=0.35,
    post_sec=0.55,
    ref_channel=REFERENCE_LEAD,
):
    start, end, r_offset, _ = aligned_beat_window_bounds(
        signals,
        channel_labels,
        sfreq,
        beat_index=beat_index,
        pre_sec=pre_sec,
        post_sec=post_sec,
        ref_channel=ref_channel,
    )
    windows = slice_aligned_leads(signals, channel_labels, sfreq, start, end)
    return windows, r_offset, start, end


def _nearest_before(values, r_idx, tol):
    cands = [int(x) for x in values if not np.isnan(x) and r_idx - int(x) <= tol and int(x) <= r_idx]
    return max(cands) if cands else None


def _nearest_after(values, r_idx, tol, after=None):
    lo = after if after is not None else r_idx
    cands = [int(x) for x in values if not np.isnan(x) and int(x) >= lo and int(x) - r_idx <= tol]
    return min(cands) if cands else None


def _landmarks_for_beat(waves, r_idx, sfreq):
    tol = int(0.55 * sfreq)

    def before(key):
        arr = waves.get(key, [])
        return _nearest_before(arr, r_idx, tol) if arr is not None else None

    def after(key, lo=None):
        arr = waves.get(key, [])
        return _nearest_after(arr, r_idx, tol, after=lo) if arr is not None else None

    p_start = before("ECG_P_Onsets") or before("ECG_P_Peaks")
    p_end = before("ECG_P_Offsets")
    if p_end is None or (p_start is not None and p_end < p_start):
        p_peak = before("ECG_P_Peaks")
        p_end = (p_peak + int(0.04 * sfreq)) if p_peak is not None else None
    if p_start is None:
        p_start = r_idx - int(0.16 * sfreq)
    if p_end is None:
        p_end = r_idx - int(0.08 * sfreq)

    qrs_start = before("ECG_Q_Peaks") or before("ECG_R_Onsets") or (r_idx - int(0.04 * sfreq))
    qrs_end = after("ECG_S_Peaks", lo=r_idx - int(0.02 * sfreq)) or (r_idx + int(0.04 * sfreq))

    st_start = qrs_end
    t_on = after("ECG_T_Onsets", lo=qrs_end) or after("ECG_T_Peaks", lo=qrs_end)
    st_end = t_on if t_on is not None else qrs_end + int(0.08 * sfreq)

    t_start = t_on if t_on is not None else st_end
    t_end = after("ECG_T_Offsets", lo=t_start) or (t_start + int(0.12 * sfreq))

    return {
        "P": (p_start, p_end),
        "QRS": (qrs_start, qrs_end),
        "ST": (st_start, st_end),
        "T": (t_start, t_end),
    }


def _clamp_interval(start, end, n):
    start = int(max(0, min(n - 1, start)))
    end = int(max(start + 1, min(n, end)))
    return start, end


def _baseline_from_segments(n, segments):
    covered = np.zeros(n, dtype=bool)
    for slc in segments.values():
        covered[slc[0] : slc[1]] = True
    baseline = []
    i = 0
    while i < n:
        if not covered[i]:
            j = i
            while j < n and not covered[j]:
                j += 1
            baseline.append((i, j))
            i = j
        else:
            i += 1
    return baseline


def segment_beats_neurokit(
    signals,
    channel_labels,
    sfreq,
    beat_index=2,
    window_start=0,
    window_end=None,
    r_offset=None,
    ref_channel=REFERENCE_LEAD,
):
    """
    Сегменты P/QRS/ST/T по neurokit на опорном отведении.
    Возвращает интервалы в координатах окна [0, window_len).
    """
    r_abs, ref = reference_r_absolute(
        signals, channel_labels, sfreq, beat_index=beat_index, ref_channel=ref_channel
    )
    clean, rpeaks = neurokit_r_peaks(ref, sfreq)
    peaks_info = {"ECG_R_Peaks": rpeaks}
    _, waves = nk.ecg_delineate(clean, peaks_info, sampling_rate=sfreq, method="prominence")

    if window_end is None:
        window_end = len(ref)
    window_len = window_end - window_start
    if r_offset is None:
        r_offset = r_abs - window_start

    landmarks = _landmarks_for_beat(waves, r_abs, sfreq)
    segments = {}
    for name, (a, b) in landmarks.items():
        segments[name] = _clamp_interval(a - window_start, b - window_start, window_len)
    baseline = _baseline_from_segments(window_len, segments)
    return segments, baseline, int(r_offset)


def process_ecg_data(signals, channel_labels, sfreq, target_channels=None, beat_index=2):
    target_channels = target_channels or TARGET_CHANNELS
    if not all(ch in channel_labels for ch in target_channels):
        missing = [ch for ch in target_channels if ch not in channel_labels]
        raise ValueError(f"Отсутствуют каналы: {missing}")

    r_abs, ref = reference_r_absolute(signals, channel_labels, sfreq, beat_index=beat_index)
    half = int(0.55 * sfreq)
    start = max(0, r_abs - half)
    end = min(len(ref), r_abs + half)
    lead_periods = slice_aligned_leads(signals, channel_labels, sfreq, start, end, target_channels)
    combined = np.hstack([lead_periods[ch] for ch in target_channels])
    return combined, lead_periods


def discretize_normalized(normalized):
    df = pd.DataFrame({"signal": [normalized.tolist()]})
    pre = Preprocessing(df.loc[0], DISCRETIZATION_FACTOR, MAX_WINDOW_SIZE)
    windows = pre.create_window()
    quantizer = discretization_lloyd_max(DISCRETIZATION_FACTOR, windows[0])
    text = pre.change_to_alphabet(quantizer, windows)[0]
    return text, quantizer


def normalize_and_discretize(combined_signal, sfreq=500):
    resampled = np.asarray(resample_signal(combined_signal, sfreq, TARGET_SFREQ), dtype=float)
    rng = resampled.max() - resampled.min()
    if rng < 1e-12:
        normalized = np.zeros_like(resampled)
    else:
        normalized = (resampled - resampled.min()) / rng

    text, quantizer = discretize_normalized(normalized)
    return normalized, text, quantizer


def lead_boundaries_in_normalized(lead_periods, target_channels=None, sfreq=500):
    """Границы отведений в нормализованном (общем) векторе — пропорционально длинам после ресемпла."""
    target_channels = target_channels or TARGET_CHANNELS
    resampled_lens = []
    for ch in target_channels:
        r = np.asarray(resample_signal(lead_periods[ch], sfreq, TARGET_SFREQ), dtype=float)
        resampled_lens.append(len(r))
    total = sum(resampled_lens)
    n_norm = sum(resampled_lens)  # приближение; фактическая длина может чуть отличаться
    bounds = [0]
    acc = 0
    for length in resampled_lens:
        acc += length
        bounds.append(acc)
    # подгонка под фактическую длину делается снаружи
    return bounds, resampled_lens


def segment_beats_per_lead(lead_period, sfreq, r_idx=None):
    """
    Эвристические фазы цикла относительно R-пика.
    Возвращает dict: имя сегмента -> (start, end) в индексах lead_period.
    r_idx: если задан — якорь R (например из extract_beat_window).
    """
    n = len(lead_period)
    if r_idx is None:
        peaks = find_r_peaks(lead_period, sfreq)
        if len(peaks):
            r_idx = int(np.argmin(np.abs(peaks - n // 2)))
        else:
            r_idx = n // 2
    else:
        r_idx = int(np.clip(r_idx, 0, n - 1))

    qrs_half = max(3, int(0.04 * sfreq))
    p_len = max(3, int(0.08 * sfreq))
    st_len = max(3, int(0.08 * sfreq))
    t_len = max(3, int(0.12 * sfreq))

    qrs = (max(0, r_idx - qrs_half), min(n, r_idx + qrs_half))
    p = (max(0, qrs[0] - p_len), qrs[0])
    st = (qrs[1], min(n, qrs[1] + st_len))
    t_wave = (st[1], min(n, st[1] + t_len))

    covered = np.zeros(n, dtype=bool)
    segments = {"P": p, "QRS": qrs, "ST": st, "T": t_wave}
    for slc in segments.values():
        covered[slc[0] : slc[1]] = True
    baseline = []
    i = 0
    while i < n:
        if not covered[i]:
            j = i
            while j < n and not covered[j]:
                j += 1
            baseline.append((i, j))
            i = j
        else:
            i += 1
    return segments, baseline, r_idx


def map_native_segments_to_normalized(segments, baseline, lead_start, lead_end, native_len):
    """Линейное отображение индексов отведения в индексы общего нормализованного вектора."""
    scale = (lead_end - lead_start) / max(native_len, 1)

    def map_range(s, e):
        return int(lead_start + s * scale), int(lead_start + e * scale)

    out_seg = {}
    for name, (s, e) in segments.items():
        if e > s:
            out_seg[name] = map_range(s, e)
    out_base = [map_range(s, e) for s, e in baseline if e > s]
    return out_seg, out_base


def discover_edf_paths(datasets_root):
    paths = {}
    for root, _, files in os.walk(datasets_root):
        for fname in files:
            if fname.lower().endswith(".edf"):
                name = os.path.splitext(fname)[0]
                paths[name] = os.path.join(root, fname)
    return paths


def load_lef_names(path):
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
