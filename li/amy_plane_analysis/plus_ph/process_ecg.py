"""
Скрипт для расчета признаков отклонения от плоскостности петель ВЭКГ
"""
import numpy as np
import pickle
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm
import os
import neurokit2 as nk
import yaml


def preprocess_ecg(x, sfreq):
    """
    Предобработка ЭКГ сигнала: приведение к нулевому среднему и применение полосового фильтра
    
    Parameters:
    -----------
    x : np.ndarray
        ЭКГ сигнал
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    np.ndarray
        Обработанный сигнал
    """
    # Приведение к нулевому среднему
    x = x - np.mean(x)
    
    # Применение полосового фильтра Баттерворта
    nyq = 0.5 * sfreq
    b, a = butter(1, [0.6 / nyq, 40 / nyq], btype="bandpass")
    x = filtfilt(b, a, x)
    
    return x


def calculate_vecg(V1, V2, V3, V4, V5, V6, I, II):
    """
    Расчет векторной ЭКГ (ВЭКГ) по отведениям
    
    Parameters:
    -----------
    V1-V6, I, II : np.ndarray
        Отведения ЭКГ
        
    Returns:
    --------
    tuple
        (x, y, z) компоненты ВЭКГ
    """
    x = -(-0.172*V1 - 0.074*V2 + 0.122*V3 + 0.231*V4 + 0.239*V5 + 0.194*V6 + 0.156*I - 0.01*II)
    y = (0.057*V1 - 0.019*V2 - 0.106*V3 - 0.022*V4 + 0.041*V5 + 0.048*V6 - 0.227*I + 0.887*II)
    z = -(-0.229*V1 - 0.31*V2 - 0.246*V3 - 0.063*V4 + 0.055*V5 + 0.108*V6 + 0.022*I + 0.102*II)
    
    return x, y, z


def detect_qrs_regions(vecg_magnitude, fs=500):
    """
    Детекция QRS комплексов по величине векторной ЭКГ с использованием neurokit2
    Находит QRS комплекс из усредненного сигнала по новому алгоритму
    """
    try:
        # находим QRS комплекс из усредненного сигнала
        _, rpeaks = nk.ecg_peaks(vecg_magnitude, sampling_rate=fs)
        signal, waves = nk.ecg_delineate(vecg_magnitude, rpeaks, sampling_rate=fs, method="dwt", show=False, show_type='all')
        peaks = rpeaks['ECG_R_Peaks']
        
        # Ищем QRS комплекс с максимальной длительностью среди первых 10
        delta = 0
        index = 0
        start = None
        stop = None
        
        q_peaks = waves.get('ECG_Q_Peaks', [])
        s_peaks = waves.get('ECG_S_Peaks', [])
        
        for i in range(min(10, len(peaks))):
            if i < len(q_peaks) and i < len(s_peaks):
                qrs_onset = q_peaks[i]
                qrs_offset = s_peaks[i]
                
                if qrs_onset is not None and qrs_offset is not None:
                    delta1 = qrs_offset - qrs_onset
                    if delta1 > delta:
                        delta = delta1
                        index = i
                        start = qrs_onset
                        stop = qrs_offset
        
        if start is None or stop is None:
            # Если не удалось найти через delineate, используем упрощенный метод
            if len(peaks) > 0:
                # Берем первый пик и используем фиксированный интервал
                peak_idx = peaks[0]
                delta = int(0.05 * fs)  # 50 мс
                start = max(0, peak_idx - delta)
                stop = min(len(vecg_magnitude) - 1, peak_idx + delta)
            else:
                return []
        
        qrs_regions = []
        qrs_start = start
        qrs_end = stop
        qrs_center = (qrs_start + qrs_end) // 2
        qrs_duration = qrs_end - qrs_start
        
        # Определяем 5 равноотстоящих точек внутри QRS комплекса
        step = max(1, qrs_duration // 4)  # 4 интервала = 5 точек
        
        analysis_points = [
            qrs_start,
            qrs_start + step,
            qrs_center,
            qrs_start + 3*step,
            qrs_end
        ]
        
        if qrs_start < qrs_end and qrs_end < len(vecg_magnitude):
            qrs_regions.append({
                'start': qrs_start,
                'center': qrs_center,
                'end': qrs_end,
                'duration': qrs_duration,
                'analysis_points': analysis_points,
                'indices': list(range(qrs_start, qrs_end + 1))
            })
        
        return qrs_regions
        
    except Exception as e:
        print(f"Ошибка при детекции QRS: {e}")
        # Резервный вариант
        min_distance = int(0.6 * fs)
        min_height = np.percentile(vecg_magnitude, 80)
        
        peaks, _ = find_peaks(vecg_magnitude, distance=min_distance, height=min_height)
        
        qrs_regions = []
        if len(peaks) > 0:
            peak_idx = peaks[0]
            delta = int(0.05 * fs)
            qrs_start = max(0, peak_idx - delta)
            qrs_end = min(len(vecg_magnitude) - 1, peak_idx + delta)
            qrs_center = (qrs_start + qrs_end) // 2
            qrs_duration = qrs_end - qrs_start
            
            step = max(1, qrs_duration // 4)
            analysis_points = [
                qrs_start,
                qrs_start + step,
                qrs_center,
                qrs_start + 3*step,
                qrs_end
            ]
            
            qrs_regions.append({
                'start': qrs_start,
                'center': qrs_center,
                'end': qrs_end,
                'duration': qrs_duration,
                'analysis_points': analysis_points,
                'indices': list(range(qrs_start, qrs_end + 1))
            })
        
        return qrs_regions


def detect_st_regions(vecg_magnitude, qrs_regions, fs=500):
    """
    Детекция ST сегментов от S до конца периода
    Использует найденные QRS для определения ST
    """
    st_regions = []
    
    try:
        # Используем neurokit2 для детекции точек S
        _, rpeaks_info = nk.ecg_peaks(vecg_magnitude, sampling_rate=fs)
        signal, waves = nk.ecg_delineate(
            vecg_magnitude,
            rpeaks_info,
            sampling_rate=fs,
            method="dwt",
            show=False,
            show_type='all'
        )
        
        # Находим ST от S до конца периода
        s_peaks = waves.get('ECG_S_Peaks', [])
        
        # Используем тот же индекс, что был найден для QRS
        if len(qrs_regions) > 0:
            # Берем первый QRS регион (из усредненного сигнала)
            qrs = qrs_regions[0]
            
            # Ищем соответствующий индекс в waves
            # Находим индекс пика, который соответствует найденному QRS
            peaks = rpeaks_info['ECG_R_Peaks']
            qrs_center = qrs['center']
            
            # Находим ближайший пик к центру QRS
            closest_peak_idx = 0
            if len(peaks) > 0:
                min_dist = abs(peaks[0] - qrs_center)
                for i, peak in enumerate(peaks):
                    dist = abs(peak - qrs_center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_peak_idx = i
            
            # Используем индекс для поиска S, конец ST - конец периода
            if closest_peak_idx < len(s_peaks):
                s_peak = s_peaks[closest_peak_idx]
                
                if s_peak is not None:
                    st_start = s_peak
                    # Конец ST - конец периода (сигнала)
                    st_end = len(vecg_magnitude) - 1
                    
                    if st_end > st_start and st_end < len(vecg_magnitude):
                        st_center = (st_start + st_end) // 2
                        st_duration = st_end - st_start
                        
                        # Определяем 5 равноотстоящих точек внутри ST сегмента
                        step = max(1, st_duration // 4)  # 4 интервала = 5 точек
                        
                        analysis_points = [
                            st_start,
                            st_start + step,
                            st_center,
                            st_start + 3*step,
                            st_end
                        ]
                        
                        st_regions.append({
                            'start': st_start,
                            'center': st_center,
                            'end': st_end,
                            'duration': st_duration,
                            'analysis_points': analysis_points,
                            'indices': list(range(st_start, st_end + 1))
                        })
        
        # Если не удалось найти через delineate, используем конец QRS как начало ST
        if not st_regions and len(qrs_regions) > 0:
            qrs = qrs_regions[0]
            # ST начинается от конца QRS (S peak)
            st_start = qrs['end']
            # ST заканчивается в конце периода
            st_end = len(vecg_magnitude) - 1
            
            if st_end > st_start:
                st_center = (st_start + st_end) // 2
                st_duration = st_end - st_start
                
                step = max(1, st_duration // 4)
                
                analysis_points = [
                    st_start,
                    st_start + step,
                    st_center,
                    st_start + 3*step,
                    st_end
                ]
                
                st_regions.append({
                    'start': st_start,
                    'center': st_center,
                    'end': st_end,
                    'duration': st_duration,
                    'analysis_points': analysis_points,
                    'indices': list(range(st_start, st_end + 1))
                })
    
    except Exception as e:
        print(f"Ошибка при детекции ST: {e}")
        # Резервный вариант
        if len(qrs_regions) > 0:
            qrs = qrs_regions[0]
            st_start = qrs['end']
            # ST заканчивается в конце периода
            st_end = len(vecg_magnitude) - 1
            
            if st_end > st_start:
                st_center = (st_start + st_end) // 2
                st_duration = st_end - st_start
                
                step = max(1, st_duration // 4)
                
                analysis_points = [
                    st_start,
                    st_start + step,
                    st_center,
                    st_start + 3*step,
                    st_end
                ]
                
                st_regions.append({
                    'start': st_start,
                    'center': st_center,
                    'end': st_end,
                    'duration': st_duration,
                    'analysis_points': analysis_points,
                    'indices': list(range(st_start, st_end + 1))
                })
    
    return st_regions


def average_periods(vecg_x, vecg_y, vecg_z, num_periods=5, sfreq=500):
    """
    Усреднение ВЭКГ по 5 периодам, начиная со второго, с использованием neurokit2
    
    Parameters:
    -----------
    vecg_x, vecg_y, vecg_z : np.ndarray
        Компоненты ВЭКГ
    num_periods : int
        Количество периодов для усреднения (должно быть 5)
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    tuple
        Усредненные компоненты ВЭГ
    """
    # Используем комбинацию всех компонент для более надежного поиска
    combined_signal = np.sqrt(vecg_x**2 + vecg_y**2 + vecg_z**2)
    
    # Searching for 1 period (для визуализации, но не используем в усреднении)
    # Находим R-пики через neurokit2
    try:
        _, rpeaks_info = nk.ecg_peaks(combined_signal, sampling_rate=sfreq)
        peaks = rpeaks_info['ECG_R_Peaks']
    except:
        # Резервный вариант
        min_distance = int(0.4 * sfreq)
        min_height = np.percentile(combined_signal, 80)
        peaks, _ = find_peaks(combined_signal, distance=min_distance, height=min_height)
        peaks = peaks.tolist()
    
    if len(peaks) < num_periods + 1:
        # Если пиков недостаточно, возвращаем исходный сигнал
        return vecg_x, vecg_y, vecg_z
    
    # Выделяем 5 периодов, начиная со второго (индексы 1-5)
    # Для первого периода: delta = round((peaks[1] - peaks[0])/2)
    periods_x = []
    periods_y = []
    periods_z = []
    
    for i in range(1, min(num_periods + 1, len(peaks))):
        # Вычисляем delta для каждого периода (половина расстояния до предыдущего пика)
        delta = round((peaks[i] - peaks[i - 1]) / 2)
        
        start = max(0, peaks[i] - delta)
        end = min(len(vecg_x), peaks[i] + delta)
        
        if end > start:
            periods_x.append(vecg_x[start:end])
            periods_y.append(vecg_y[start:end])
            periods_z.append(vecg_z[start:end])
    
    if len(periods_x) == 0:
        # Если не удалось выделить периоды, возвращаем исходный сигнал
        return vecg_x, vecg_y, vecg_z
    
    # Приводим все периоды к одинаковой длине через интерполяцию
    target_length = int(np.median([len(p) for p in periods_x]))
    
    interpolated_periods_x = []
    interpolated_periods_y = []
    interpolated_periods_z = []
    
    for i in range(len(periods_x)):
        old_indices = np.linspace(0, len(periods_x[i]) - 1, len(periods_x[i]))
        new_indices = np.linspace(0, len(periods_x[i]) - 1, target_length)
        
        f_x = interp1d(old_indices, periods_x[i], kind='linear', fill_value='extrapolate')
        f_y = interp1d(old_indices, periods_y[i], kind='linear', fill_value='extrapolate')
        f_z = interp1d(old_indices, periods_z[i], kind='linear', fill_value='extrapolate')
        
        interpolated_periods_x.append(f_x(new_indices))
        interpolated_periods_y.append(f_y(new_indices))
        interpolated_periods_z.append(f_z(new_indices))
    
    # Усредняем периоды (среднее арифметическое)
    avg_x = np.mean(interpolated_periods_x, axis=0)
    avg_y = np.mean(interpolated_periods_y, axis=0)
    avg_z = np.mean(interpolated_periods_z, axis=0)
    
    return avg_x, avg_y, avg_z


def extract_qrs_loop(vecg_x, vecg_y, vecg_z, sfreq=500):
    """
    Выделение петли QRS из ВЭКГ с использованием neurokit2
    
    Parameters:
    -----------
    vecg_x, vecg_y, vecg_z : np.ndarray
        Компоненты ВЭКГ
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    tuple
        (x, y, z) компоненты петли QRS
    """
    # Используем комбинацию всех компонент для поиска
    combined_signal = np.sqrt(vecg_x**2 + vecg_y**2 + vecg_z**2)
    
    # Находим QRS регионы
    qrs_regions = detect_qrs_regions(combined_signal, fs=sfreq)
    
    if len(qrs_regions) == 0:
        # Если регионы не найдены, используем максимальное значение
        max_idx = np.argmax(combined_signal)
        qrs_duration_ms = 150
        window_size = int(qrs_duration_ms * sfreq / 1000)
        window_size = min(window_size, len(vecg_x) // 3)
        start = max(0, max_idx - window_size // 2)
        end = min(len(vecg_x), max_idx + window_size // 2)
        return vecg_x[start:end], vecg_y[start:end], vecg_z[start:end]
    
    # Берем первый QRS регион и объединяем все точки
    qrs_region = qrs_regions[0]
    indices = qrs_region['indices']
    
    # Если есть несколько регионов, можем усреднить или взять самый большой
    if len(qrs_regions) > 1:
        # Берем регион с наибольшей длительностью
        qrs_region = max(qrs_regions, key=lambda x: x['duration'])
        indices = qrs_region['indices']
    
    # Извлекаем петлю по индексам
    valid_indices = [i for i in indices if 0 <= i < len(vecg_x)]
    if len(valid_indices) == 0:
        # Fallback
        max_idx = np.argmax(combined_signal)
        window_size = min(75, len(vecg_x) // 3)
        start = max(0, max_idx - window_size // 2)
        end = min(len(vecg_x), max_idx + window_size // 2)
        return vecg_x[start:end], vecg_y[start:end], vecg_z[start:end]
    
    return vecg_x[valid_indices], vecg_y[valid_indices], vecg_z[valid_indices]


def extract_st_loop(vecg_x, vecg_y, vecg_z, sfreq=500):
    """
    Выделение петли ST из ВЭКГ с использованием neurokit2
    
    Parameters:
    -----------
    vecg_x, vecg_y, vecg_z : np.ndarray
        Компоненты ВЭКГ
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    tuple
        (x, y, z) компоненты петли ST
    """
    # Используем комбинацию всех компонент для поиска
    combined_signal = np.sqrt(vecg_x**2 + vecg_y**2 + vecg_z**2)
    
    # Находим QRS регионы
    qrs_regions = detect_qrs_regions(combined_signal, fs=sfreq)
    
    if len(qrs_regions) == 0:
        # Если регионы не найдены, используем фиксированный интервал
        max_idx = np.argmax(combined_signal)
        st_start = max_idx + int(0.07 * sfreq)
        st_end = st_start + int(0.11 * sfreq)
        if st_end >= len(vecg_x):
            st_end = len(vecg_x) - 1
        if st_start >= st_end:
            st_start = max(0, len(vecg_x) // 2)
            st_end = min(len(vecg_x), st_start + int(0.11 * sfreq))
        return vecg_x[st_start:st_end], vecg_y[st_start:st_end], vecg_z[st_start:st_end]
    
    # Находим ST регионы
    st_regions = detect_st_regions(combined_signal, qrs_regions, fs=sfreq)
    
    if len(st_regions) == 0:
        # Если ST регионы не найдены, используем фиксированный интервал после первого QRS
        qrs = qrs_regions[0]
        st_start = qrs['end'] + int(0.07 * sfreq)
        st_end = st_start + int(0.11 * sfreq)
        if st_end >= len(vecg_x):
            st_end = len(vecg_x) - 1
        if st_start >= st_end:
            st_start = max(0, qrs['end'])
            st_end = min(len(vecg_x), st_start + int(0.11 * sfreq))
        return vecg_x[st_start:st_end], vecg_y[st_start:st_end], vecg_z[st_start:st_end]
    
    # Берем первый ST регион и объединяем все точки
    st_region = st_regions[0]
    indices = st_region['indices']
    
    # Если есть несколько регионов, берем самый большой
    if len(st_regions) > 1:
        st_region = max(st_regions, key=lambda x: x['duration'])
        indices = st_region['indices']
    
    # Извлекаем петлю по индексам
    valid_indices = [i for i in indices if 0 <= i < len(vecg_x)]
    if len(valid_indices) == 0:
        # Fallback
        qrs = qrs_regions[0]
        st_start = qrs['end'] + int(0.07 * sfreq)
        st_end = st_start + int(0.11 * sfreq)
        if st_end >= len(vecg_x):
            st_end = len(vecg_x) - 1
        return vecg_x[st_start:st_end], vecg_y[st_start:st_end], vecg_z[st_start:st_end]
    
    return vecg_x[valid_indices], vecg_y[valid_indices], vecg_z[valid_indices]


def calculate_planarity_r2(x, y, z):
    """
    Расчет R² для отклонения от плоскостности методом МНК
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Компоненты петли
        
    Returns:
    --------
    float
        R² значение
    """
    if len(x) < 3:
        return 0.0
    
    # Проверка на константность (все точки одинаковые)
    if np.std(x) < 1e-10 and np.std(y) < 1e-10 and np.std(z) < 1e-10:
        return 1.0  # Все точки совпадают - идеальная плоскость
    
    # Проверка на коллинеарность в плоскости XY
    # Если все точки лежат на одной прямой в плоскости XY, то z должен быть константой
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        # Если x или y константны, проверяем, насколько z отклоняется от константы
        if np.std(z) < 1e-10:
            return 1.0  # Все точки лежат на одной прямой, z константа
        else:
            # z не константа, но x или y константны - это не плоскость
            return 0.0
    
    # Подготавливаем данные для регрессии
    # Ищем плоскость z = ax + by + c
    X = np.column_stack([x, y, np.ones(len(x))])
    Z = z
    
    # МНК регрессия через псевдообратную матрицу (более устойчиво)
    try:
        # Используем SVD для более устойчивого решения
        coeffs, residuals, rank, s = np.linalg.lstsq(X, Z, rcond=None)
        
        # Проверяем, что решение найдено
        if len(coeffs) == 0:
            return 0.0
        
        # Предсказанные значения
        z_pred = X @ coeffs
        
        # R² через формулу: 1 - (SS_res / SS_tot)
        ss_res = np.sum((Z - z_pred) ** 2)
        ss_tot = np.sum((Z - np.mean(Z)) ** 2)
        
        if ss_tot < 1e-10:
            # Если z константа, то идеальное предсказание
            return 1.0
        
        r2 = 1.0 - (ss_res / ss_tot)
        
        # Проверяем на NaN и Inf
        if np.isnan(r2) or np.isinf(r2):
            return 0.0
        
        # Ограничиваем R² в диапазоне [0, 1]
        r2 = max(0.0, min(1.0, r2))
        
        return r2
    except:
        # Если не удалось решить, используем sklearn
        try:
            reg = LinearRegression()
            reg.fit(X[:, :2], Z)
            z_pred = reg.predict(X[:, :2])
            r2 = r2_score(Z, z_pred)
            
            # Проверяем на NaN и Inf
            if np.isnan(r2) or np.isinf(r2):
                return 0.0
            
            return max(0.0, min(1.0, r2))
        except:
            return 0.0


def calculate_r2_for_point_with_expanding_regions(x, y, z, point_idx, num_regions=5):
    """
    Расчет R² для точки с расширяющимися окрестностями
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Компоненты петли
    point_idx : int
        Индекс центральной точки
    num_regions : int
        Количество расширяющихся окрестностей (по умолчанию 5)
        
    Returns:
    --------
    list
        Список R² для каждой окрестности (от малой к большой)
    """
    if len(x) < 3:
        return [0.0] * num_regions
    
    # Ограничиваем индекс точки
    point_idx = max(0, min(point_idx, len(x) - 1))
    
    r2_list = []
    
    # Вычисляем размеры окрестностей
    # Последняя окрестность должна охватывать всю петлю
    loop_length = len(x)
    
    # Размеры окрестностей: от малой до полной петли
    # Используем геометрическую прогрессию для плавного расширения
    for region_idx in range(num_regions):
        # Доля петли, которую должна охватывать окрестность
        # От 0.1 (10%) до 1.0 (100%) для последней окрестности
        fraction = 0.1 + (region_idx / (num_regions - 1)) * 0.9 if num_regions > 1 else 1.0
        
        # Размер окрестности в точках
        region_size = max(3, int(loop_length * fraction))
        
        # Определяем границы окрестности с центром в point_idx
        half_size = region_size // 2
        start = max(0, point_idx - half_size)
        end = min(len(x), point_idx + half_size + 1)
        
        # Если окрестность слишком маленькая, расширяем
        if end - start < 3:
            if start == 0:
                end = min(len(x), start + 3)
            elif end == len(x):
                start = max(0, end - 3)
            else:
                # Расширяем в обе стороны
                start = max(0, point_idx - 1)
                end = min(len(x), point_idx + 2)
        
        # Для последней окрестности берем всю петлю
        if region_idx == num_regions - 1:
            start = 0
            end = len(x)
        
        # Вычисляем R² для этой окрестности
        if end - start >= 3:
            r2 = calculate_planarity_r2(x[start:end], y[start:end], z[start:end])
            # Проверяем на NaN и Inf
            if np.isnan(r2) or np.isinf(r2):
                r2 = 0.0
            r2_list.append(r2)
        else:
            # Если окрестность слишком маленькая, используем предыдущее значение или 0
            if len(r2_list) > 0:
                r2_list.append(r2_list[-1])
            else:
                r2_list.append(0.0)
    
    return r2_list


def calculate_r2_for_multiple_points(x, y, z, num_points=5, num_regions_per_point=5):
    """
    Расчет R² для нескольких точек с расширяющимися окрестностями
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Компоненты петли
    num_points : int
        Количество точек на петле (по умолчанию 5)
    num_regions_per_point : int
        Количество окрестностей для каждой точки (по умолчанию 5)
        
    Returns:
    --------
    dict
        Словарь с R² для каждой точки и каждой окрестности
        Формат: {'point_1': [r2_region_1, r2_region_2, ...], ...}
    """
    if len(x) < 3:
        return {f'point_{i+1}': [0.0] * num_regions_per_point for i in range(num_points)}
    
    results = {}
    
    # Определяем равномерно распределенные точки на петле
    if len(x) >= num_points:
        # Равномерно распределенные индексы
        point_indices = [int(i * (len(x) - 1) / (num_points - 1)) for i in range(num_points)]
    else:
        # Если точек меньше, чем нужно, используем все доступные
        point_indices = list(range(len(x)))
        # Дополняем до num_points, повторяя последнюю точку
        while len(point_indices) < num_points:
            point_indices.append(len(x) - 1)
        point_indices = point_indices[:num_points]
    
    # Для каждой точки вычисляем R² для всех окрестностей
    for i, point_idx in enumerate(point_indices):
        r2_list = calculate_r2_for_point_with_expanding_regions(
            x, y, z, point_idx, num_regions=num_regions_per_point
        )
        results[f'point_{i+1}'] = r2_list
    
    return results 


def process_patient(ecg_data, sfreq, num_periods, num_regions):
    """
    Обработка данных одного пациента
    
    Parameters:
    -----------
    ecg_data : dict or np.ndarray
        Данные ЭКГ пациента
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    dict
        Словарь с признаками
    """
    # Извлекаем отведения
    if isinstance(ecg_data, dict):
        V1 = ecg_data.get('V1', ecg_data.get('v1'))
        V2 = ecg_data.get('V2', ecg_data.get('v2'))
        V3 = ecg_data.get('V3', ecg_data.get('v3'))
        V4 = ecg_data.get('V4', ecg_data.get('v4'))
        V5 = ecg_data.get('V5', ecg_data.get('v5'))
        V6 = ecg_data.get('V6', ecg_data.get('v6'))
        I = ecg_data.get('I', ecg_data.get('i'))
        II = ecg_data.get('II', ecg_data.get('ii'))
    elif isinstance(ecg_data, np.ndarray):
        # Предполагаем, что данные в формате [channels, time] или [time, channels]
        if ecg_data.ndim == 2:
            if ecg_data.shape[0] > ecg_data.shape[1]:
                # [time, channels]
                ecg_data = ecg_data.T
            # Теперь [channels, time]
            # Предполагаем порядок: I, II, V1, V2, V3, V4, V5, V6
            if ecg_data.shape[0] >= 8:
                I = ecg_data[0]
                II = ecg_data[1]
                V1 = ecg_data[2]
                V2 = ecg_data[3]
                V3 = ecg_data[4]
                V4 = ecg_data[5]
                V5 = ecg_data[6]
                V6 = ecg_data[7]
            else:
                raise ValueError(f"Недостаточно каналов: {ecg_data.shape[0]}")
        else:
            raise ValueError(f"Неожиданная размерность данных: {ecg_data.ndim}")
    else:
        raise ValueError(f"Неожиданный тип данных: {type(ecg_data)}")
    
    # Предобработка каждого отведения
    V1 = preprocess_ecg(V1, sfreq)
    V2 = preprocess_ecg(V2, sfreq)
    V3 = preprocess_ecg(V3, sfreq)
    V4 = preprocess_ecg(V4, sfreq)
    V5 = preprocess_ecg(V5, sfreq)
    V6 = preprocess_ecg(V6, sfreq)
    I = preprocess_ecg(I, sfreq)
    II = preprocess_ecg(II, sfreq)
    
    # Расчет ВЭКГ
    vecg_x, vecg_y, vecg_z = calculate_vecg(V1, V2, V3, V4, V5, V6, I, II)
    
    # Усреднение по периодам
    avg_x, avg_y, avg_z = average_periods(vecg_x, vecg_y, vecg_z, num_periods=num_periods, sfreq=sfreq)
    
    # Выделение петель
    qrs_x, qrs_y, qrs_z = extract_qrs_loop(avg_x, avg_y, avg_z, sfreq=sfreq)
    st_x, st_y, st_z = extract_st_loop(avg_x, avg_y, avg_z, sfreq=sfreq)
    
    # Проверка минимальной длины петель
    min_loop_length = 10
    if len(qrs_x) < min_loop_length:
        # Если QRS слишком короткая, используем весь сигнал вокруг максимума
        combined = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
        max_idx = np.argmax(combined)
        window = min(min_loop_length * 2, len(avg_x) // 4)
        start = max(0, max_idx - window // 2)
        end = min(len(avg_x), max_idx + window // 2)
        qrs_x, qrs_y, qrs_z = avg_x[start:end], avg_y[start:end], avg_z[start:end]
    
    if len(st_x) < min_loop_length:
        # Если ST слишком короткая, используем интервал после QRS
        combined = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
        max_idx = np.argmax(combined)
        st_start = max_idx + int(0.08 * sfreq)  # 80 мс после QRS
        st_end = st_start + int(0.12 * sfreq)   # 120 мс длительность
        if st_end >= len(avg_x):
            st_end = len(avg_x) - 1
        if st_start < st_end:
            st_x, st_y, st_z = avg_x[st_start:st_end], avg_y[st_start:st_end], avg_z[st_start:st_end]
        else:
            # Если не получилось, используем последнюю часть сигнала
            st_start = max(0, len(avg_x) - int(0.15 * sfreq))
            st_end = len(avg_x)
            st_x, st_y, st_z = avg_x[st_start:st_end], avg_y[st_start:st_end], avg_z[st_start:st_end]
    
    # Расчет признаков для QRS
    # Total R² для всей петли (один раз)
    qrs_r2_total = calculate_planarity_r2(qrs_x, qrs_y, qrs_z)
    
    # R² для 5 точек с расширяющимися окрестностями
    qrs_r2_points = calculate_r2_for_multiple_points(
        qrs_x, qrs_y, qrs_z, 
        num_points=5, 
        num_regions_per_point=5
    )
    
    # Расчет признаков для ST
    # Total R² для всей петли (один раз)
    st_r2_total = calculate_planarity_r2(st_x, st_y, st_z)
    
    # R² для 5 точек с расширяющимися окрестностями
    st_r2_points = calculate_r2_for_multiple_points(
        st_x, st_y, st_z, 
        num_points=5, 
        num_regions_per_point=5
    )
    
    # Формирование результата
    features = {
        'QRS_R2_total': qrs_r2_total,
    }
    
    # Добавляем признаки для каждой точки QRS
    for point_name, r2_list in qrs_r2_points.items():
        point_num = point_name.split('_')[1]  # 'point_1' -> '1'
        for region_idx, r2 in enumerate(r2_list, start=1):
            features[f'QRS_R2_point_{point_num}_region_{region_idx}'] = r2
    
    # Добавляем total для ST
    features['ST_R2_total'] = st_r2_total
    
    # Добавляем признаки для каждой точки ST
    for point_name, r2_list in st_r2_points.items():
        point_num = point_name.split('_')[1]  # 'point_1' -> '1'
        for region_idx, r2 in enumerate(r2_list, start=1):
            features[f'ST_R2_point_{point_num}_region_{region_idx}'] = r2
    
    return features


def process_dataset(dataset_path, output_path=None, sfreq=500, num_periods=5, num_regions=5):
    """
    Обработка всего датасета
    
    Parameters:
    -----------
    dataset_path : str
        Путь к файлу с данными
    output_path : str
        Путь для сохранения результатов
    sfreq : float
        Частота дискретизации
        
    Returns:
    --------
    pd.DataFrame
        Таблица с признаками
    """
    print(f"Загрузка данных из {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Тип данных: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Форма данных: {data.shape}")
    elif isinstance(data, (list, tuple)):
        print(f"Количество пациентов: {len(data)}")
    elif isinstance(data, dict):
        print(f"Ключи: {list(data.keys())}")
    
    # Обработка каждого пациента
    all_features = []
    
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            # [patients, channels, time] или [patients, time, channels]
            if data.shape[1] > data.shape[2]:
                # [patients, time, channels] -> транспонируем
                data = np.transpose(data, (0, 2, 1))
            # Теперь [patients, channels, time]
            num_patients = data.shape[0]
            for i in tqdm(range(num_patients), desc="Обработка пациентов"):
                try:
                    features = process_patient(data[i], sfreq=sfreq, num_periods=num_periods, num_regions=num_regions)
                    all_features.append(features)
                except Exception as e:
                    print(f"Ошибка при обработке пациента {i}: {e}")
                    # Добавляем нулевые признаки (1 total + 5 точек * 5 регионов для QRS и ST)
                    default_features = {
                        'QRS_R2_total': 0.0,
                        'ST_R2_total': 0.0
                    }
                    for point in range(1, 6):
                        for region in range(1, 6):
                            default_features[f'QRS_R2_point_{point}_region_{region}'] = 0.0
                            default_features[f'ST_R2_point_{point}_region_{region}'] = 0.0
                    all_features.append(default_features)
        else:
            raise ValueError(f"Неожиданная размерность массива: {data.ndim}")
    elif isinstance(data, (list, tuple)):
        for i, patient_data in enumerate(tqdm(data, desc="Обработка пациентов")):
            try:
                features = process_patient(patient_data, sfreq=sfreq, num_periods=num_periods, num_regions=num_regions)
                all_features.append(features)
            except Exception as e:
                print(f"Ошибка при обработке пациента {i}: {e}")
                # Добавляем нулевые признаки
                default_features = {
                    'QRS_R2_total': 0.0,
                    'ST_R2_total': 0.0
                }
                for point in range(1, 6):
                    for region in range(1, 6):
                        default_features[f'QRS_R2_point_{point}_region_{region}'] = 0.0
                        default_features[f'ST_R2_point_{point}_region_{region}'] = 0.0
                all_features.append(default_features)
    elif isinstance(data, dict):
        # Предполагаем, что ключи - это индексы пациентов или имена
        for key, patient_data in tqdm(data.items(), desc="Обработка пациентов"):
            try:
                features = process_patient(patient_data, sfreq=sfreq, num_periods=num_periods, num_regions=num_regions)
                all_features.append(features)
            except Exception as e:
                print(f"Ошибка при обработке пациента {key}: {e}")
                # Добавляем нулевые признаки
                default_features = {
                    'QRS_R2_total': 0.0,
                    'ST_R2_total': 0.0
                }
                for point in range(1, 6):
                    for region in range(1, 6):
                        default_features[f'QRS_R2_point_{point}_region_{region}'] = 0.0
                        default_features[f'ST_R2_point_{point}_region_{region}'] = 0.0
                all_features.append(default_features)
    
    # Создание DataFrame
    df = pd.DataFrame(all_features)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Результаты сохранены в {output_path}")
    
    return df


def load_config(config_path="config.yaml"):
    """
    Загрузка конфигурации из YAML файла
    
    Parameters:
    -----------
    config_path : str
        Путь к файлу конфигурации
        
    Returns:
    --------
    dict
        Словарь с конфигурацией
    """
    if not os.path.exists(config_path):
        print(f"Файл конфигурации {config_path} не найден. Используются значения по умолчанию.")
        return {
            'dataset_dir': "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset",
            'dataset_type': "original",
            'sampling_frequency': 500,
            'features_dir': "features",
            'processing': {
                'num_periods': 5,
                'num_regions': 5
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    # Загружаем конфигурацию
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)
    
    dataset_dir = config.get('dataset_dir', "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset")
    dataset_type = config.get('dataset_type')
    sfreq = config.get('sampling_frequency')
    features_dir = config.get('features_dir')
    num_periods = config.get('processing', {}).get('num_periods')
    num_regions = config.get('processing', {}).get('num_regions')
    
    # Определяем пути к файлам в зависимости от типа датасета
    if dataset_type == "stratified":
        train_filename = "X_train_stratified.pkl"
        test_filename = "X_test_stratified.pkl"
        print(f"Используется стратифицированный датасет")
    else:
        train_filename = "X_train.pkl"
        test_filename = "X_test.pkl"
        print(f"Используется исходный датасет")
    
    train_path = os.path.join(dataset_dir, train_filename)
    test_path = os.path.join(dataset_dir, test_filename)
    
    # Проверяем существование файлов
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Файл {train_path} не найден!")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Файл {test_path} не найден!")
    
    # Создаем папку для признаков (абсолютный путь или относительно скрипта)
    if os.path.isabs(features_dir):
        features_dir_abs = features_dir
    else:
        features_dir_abs = os.path.join(script_dir, features_dir)
    os.makedirs(features_dir_abs, exist_ok=True)
    
    print(f"\nКонфигурация:")
    print(f"  Директория данных: {dataset_dir}")
    print(f"  Тип датасета: {dataset_type}")
    print(f"  Частота дискретизации: {sfreq} Гц")
    print(f"  Директория признаков: {features_dir_abs}")
    print(f"  Количество периодов: {num_periods}")
    print(f"  Количество регионов: {num_regions}")
    
    print("\n" + "="*60)
    print("Обработка обучающего датасета...")
    print("="*60)
    train_features = process_dataset(train_path, 
                                     output_path=os.path.join(features_dir_abs, "train_features.csv"),
                                     sfreq=sfreq,
                                     num_periods=num_periods,
                                     num_regions=num_regions)
    
    print("\n" + "="*60)
    print("Обработка тестового датасета...")
    print("="*60)
    test_features = process_dataset(test_path,
                                    output_path=os.path.join(features_dir_abs, "test_features.csv"),
                                    sfreq=sfreq,
                                    num_periods=num_periods,
                                    num_regions=num_regions)
    
    print("\n" + "="*60)
    print("Готово!")
    print("="*60)
    print(f"Обучающий датасет: {train_features.shape}")
    print(f"Тестовый датасет: {test_features.shape}")
    print(f"\nПризнаки: {list(train_features.columns)}")

