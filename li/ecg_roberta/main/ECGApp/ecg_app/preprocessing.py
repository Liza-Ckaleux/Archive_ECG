import numpy as np
import pandas as pd
import neurokit2 as nk
import math
import matplotlib.pyplot as plt
import logging
from django.contrib import messages

logger = logging.getLogger(__name__)

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
        result = self._quantize_with_breakpoints(tmp)
        return result

    def learn_breakpoints(self, arr):
        res = self.preprocess(arr)
        sorted_array = np.sort(res)
        length = len(sorted_array)
        probs = [1 / self.alphabet_size for _ in range(self.alphabet_size)]
        cum_sum_breakpoints = [int(sum(probs[0:i + 1]) * length - 1) for i in range(len(probs))]
        cum_sum_breakpoint_values = sorted_array[cum_sum_breakpoints]
        cum_sum_breakpoint_values[-1] = 1e+100
        self.breakpoints = cum_sum_breakpoint_values
        return cum_sum_breakpoint_values

    @staticmethod
    def _breakpoint_to_letter(float_num, breakpoints):
        int_val = next((breakpoints.index(obj) for obj in breakpoints if float_num < obj))
        return chr(65 + int_val)

    def _quantize_with_breakpoints(self, tmp):
        breakpoints = self.breakpoints
        vect_breakpoint_to_letter = np.vectorize(self._breakpoint_to_letter, excluded=['breakpoints'])
        tmp = vect_breakpoint_to_letter(tmp, breakpoints=list(breakpoints))
        return tmp

    @staticmethod
    def _average_over_n(tmp, n):
        return np.array([np.average(tmp[i:i + n]) for i in range(0, len(tmp), n)])

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
        mlen = 0
        ecg_list = self.input_file['signal']
        if len(ecg_list) > mlen:
            mlen = len(ecg_list)

        if mlen > self.max_window_size:
            window_size = self.max_window_size
        else:
            window_size = mlen

        num_lines = math.floor(len(ecg_list)/window_size)

        for i in range(num_lines):
            tmp_list = ecg_list[i*window_size:(i+1)*window_size]
            ecg_window.append(tmp_list)

        return ecg_window

    def change_to_alphabet(self, quantizer, normalized_list):
        qtz_signal = []
        qtz = quantizer
        for i in range(len(normalized_list)):
            r = qtz.perform_quantization(np.array(normalized_list[i]), breakpoints=qtz.breakpoints)
            qtz_signal.append(''.join(r))
        return qtz_signal

def resample_signal(signal, original_sampling_rate=500, target_sampling_rate=250):
    """Resample signal to target sampling rate"""
    return nk.signal_resample(signal, sampling_rate=original_sampling_rate, 
                            desired_sampling_rate=target_sampling_rate, method="FFT")

def discretization_lloyd_max(discretizition_factor, total_data):
    qtz = DiffQuantizer(alphabet_size=discretizition_factor, breakpoints=None, use_diffs=False)
    qtz.learn_breakpoints(np.array(total_data))
    return qtz

def plot_signal(signal, sfreq=None, title='Сигнал после предобработки'):
    plt.figure(figsize=(12, 4))
    if sfreq is not None:
        x = np.arange(len(signal)) / sfreq
        plt.xlabel('Время (сек)')
    else:
        x = np.arange(len(signal))
        plt.xlabel('Индекс отсчёта')
    plt.plot(x, signal)
    plt.title(title)
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def preprocess_ecg_signal(signal, max_window_size=4000, discretizition_factor=100, plot=False, sfreq=None):
    """Main preprocessing function for ECG signal"""
    # Resample signal
    resampled_signal = resample_signal(signal)
    
    # Min-max нормализация
    resampled_signal = (resampled_signal - np.min(resampled_signal)) / (np.max(resampled_signal) - np.min(resampled_signal))
    
    # Create DataFrame for preprocessing
    df = pd.DataFrame({'signal': [resampled_signal]})
    
    # Create windows
    pre = Preprocessing(df.loc[0], discretizition_factor, max_window_size)
    windows = pre.create_window()
    
    # Get total data for quantization
    total_data = windows[0]
    
    # Create quantizer
    quantizer = discretization_lloyd_max(discretizition_factor, total_data)
    
    # Convert to alphabet
    result = pre.change_to_alphabet(quantizer, windows)
    
    return result[0]  # Return first (and only) result 

def extract_periods(signal, sfreq, num_periods=1, distance_sec=0.6):
    peaks, _ = find_peaks(signal, distance=int(distance_sec * sfreq * 0.8))
    logger.info(f'Найдено пиков: {len(peaks)}')
    if len(peaks) < num_periods + 1:
        logger.error('Недостаточно пиков для выделения периода')
        raise ValueError("Недостаточно пиков")
    periods = []
    for i in range(num_periods):
        start = peaks[i]
        end = peaks[i+1]
        half = int((end-start)/2)
        logger.info(f'Период {i}: start={start}, end={end}, half={half}')
        periods.append(signal[end-half:end+half])
    return periods 