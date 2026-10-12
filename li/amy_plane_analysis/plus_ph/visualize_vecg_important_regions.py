"""
Скрипт для визуализации ВЭКГ с выделением 5 наиболее важных участков
на основе feature importance из Random Forest
"""
import os
import sys
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import pandas as pd
import seaborn as sns

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_ecg import (
    preprocess_ecg, calculate_vecg, average_periods,
    detect_qrs_regions, detect_st_regions, extract_qrs_loop, extract_st_loop
)
from classification.utils import get_feature_importance
from classification.data_loader import load_config


def load_patient_data(patient_idx, config):
    """Загрузка данных одного пациента"""
    dataset_dir = config.get('dataset_dir')
    dataset_type = config.get('dataset_type', 'original')
    
    if dataset_type == "stratified":
        train_filename = "X_train_stratified.pkl"
    else:
        train_filename = "X_train.pkl"
    
    train_path = os.path.join(dataset_dir, train_filename)
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Файл {train_path} не найден!")
    
    print(f"Загрузка данных пациента {patient_idx} из {train_path}...")
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    
    # Извлекаем данные пациента
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            if data.shape[1] > data.shape[2]:
                data = np.transpose(data, (0, 2, 1))
            patient_data = data[patient_idx]
        else:
            raise ValueError(f"Неожиданная размерность массива: {data.ndim}")
    elif isinstance(data, (list, tuple)):
        patient_data = data[patient_idx]
    elif isinstance(data, dict):
        keys = list(data.keys())
        patient_data = data[keys[patient_idx]]
    else:
        raise ValueError(f"Неожиданный тип данных: {type(data)}")
    
    return patient_data


def extract_ecg_leads(patient_data):
    """Извлечение отведений ЭКГ из данных пациента"""
    if isinstance(patient_data, dict):
        V1 = patient_data.get('V1', patient_data.get('v1'))
        V2 = patient_data.get('V2', patient_data.get('v2'))
        V3 = patient_data.get('V3', patient_data.get('v3'))
        V4 = patient_data.get('V4', patient_data.get('v4'))
        V5 = patient_data.get('V5', patient_data.get('v5'))
        V6 = patient_data.get('V6', patient_data.get('v6'))
        I = patient_data.get('I', patient_data.get('i'))
        II = patient_data.get('II', patient_data.get('ii'))
    elif isinstance(patient_data, np.ndarray):
        if patient_data.ndim == 2:
            if patient_data.shape[0] > patient_data.shape[1]:
                patient_data = patient_data.T
            if patient_data.shape[0] >= 8:
                I = patient_data[0]
                II = patient_data[1]
                V1 = patient_data[2]
                V2 = patient_data[3]
                V3 = patient_data[4]
                V4 = patient_data[5]
                V5 = patient_data[6]
                V6 = patient_data[7]
            else:
                raise ValueError(f"Недостаточно каналов: {patient_data.shape[0]}")
        else:
            raise ValueError(f"Неожиданная размерность данных: {patient_data.ndim}")
    else:
        raise ValueError(f"Неожиданный тип данных: {type(patient_data)}")
    
    return V1, V2, V3, V4, V5, V6, I, II


def get_important_regions(model, feature_names, top_n=5):
    """Получение топ-N наиболее важных регионов из модели"""
    importances = get_feature_importance(model, feature_names)
    return get_important_regions_from_importances(importances, feature_names, top_n)


def get_important_regions_from_importances(importances, feature_names, top_n=5):
    """Получение топ-N наиболее важных регионов из массива важности"""
    # Сортируем по важности
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    important_regions = []
    for idx in top_indices:
        feature_name = feature_names[idx]
        importance = importances[idx]
        
        # Парсим название признака
        # Формат: QRS_R2_point_{point}_region_{region} или ST_R2_point_{point}_region_{region}
        parts = feature_name.split('_')
        if len(parts) >= 5:
            segment_type = parts[0]  # QRS или ST
            point_num = int(parts[3]) if parts[3].isdigit() else None
            region_num = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None
            
            important_regions.append({
                'feature_name': feature_name,
                'importance': importance,
                'segment_type': segment_type,
                'point_num': point_num,
                'region_num': region_num
            })
        elif feature_name.endswith('_total'):
            # Total признак
            segment_type = feature_name.split('_')[0]
            important_regions.append({
                'feature_name': feature_name,
                'importance': importance,
                'segment_type': segment_type,
                'point_num': None,
                'region_num': None
            })
    
    return important_regions


def find_regions_in_vecg(vecg_x, vecg_y, vecg_z, important_regions, sfreq=500):
    """Поиск соответствующих регионов в ВЭКГ сигнале с учетом расширяющихся окрестностей"""
    combined_signal = np.sqrt(vecg_x**2 + vecg_y**2 + vecg_z**2)
    
    # Находим QRS и ST регионы
    qrs_regions = detect_qrs_regions(combined_signal, fs=sfreq)
    st_regions = detect_st_regions(combined_signal, qrs_regions, fs=sfreq)
    
    found_regions = []
    
    for region_info in important_regions:
        segment_type = region_info['segment_type']
        point_num = region_info['point_num']
        region_num = region_info['region_num']
        
        if segment_type == 'QRS' and len(qrs_regions) > 0:
            qrs = qrs_regions[0]
            if point_num is not None and region_num is not None:
                # Находим конкретную точку и регион с расширяющимися окрестностями
                analysis_points = qrs.get('analysis_points', [])
                if point_num <= len(analysis_points) and point_num > 0:
                    point_idx = analysis_points[point_num - 1]
                    loop_length = qrs['end'] - qrs['start']
                    
                    # Вычисляем размер окрестности по той же логике, что и в process_ecg.py
                    # region_num от 1 до 5 (индексы 0-4)
                    region_idx = region_num - 1
                    num_regions = 5
                    
                    # Доля петли, которую должна охватывать окрестность
                    # От 0.1 (10%) до 1.0 (100%) для последней окрестности
                    fraction = 0.1 + (region_idx / (num_regions - 1)) * 0.9 if num_regions > 1 else 1.0
                    
                    # Размер окрестности в точках
                    region_size = max(3, int(loop_length * fraction))
                    
                    # Определяем границы окрестности с центром в point_idx
                    half_size = region_size // 2
                    start = max(qrs['start'], point_idx - half_size)
                    end = min(qrs['end'], point_idx + half_size + 1)
                    
                    # Для последней окрестности берем всю петлю
                    if region_idx == num_regions - 1:
                        start = qrs['start']
                        end = qrs['end']
                    
                    if end > start:
                        found_regions.append({
                            'region_info': region_info,
                            'start': start,
                            'end': end,
                            'center': point_idx,
                            'x': vecg_x[start:end],
                            'y': vecg_y[start:end],
                            'z': vecg_z[start:end]
                        })
            else:
                # Total регион - весь QRS
                found_regions.append({
                    'region_info': region_info,
                    'start': qrs['start'],
                    'end': qrs['end'],
                    'center': qrs['center'],
                    'x': vecg_x[qrs['start']:qrs['end']],
                    'y': vecg_y[qrs['start']:qrs['end']],
                    'z': vecg_z[qrs['start']:qrs['end']]
                })
        
        elif segment_type == 'ST' and len(st_regions) > 0:
            st = st_regions[0]
            if point_num is not None and region_num is not None:
                # Находим конкретную точку и регион с расширяющимися окрестностями
                analysis_points = st.get('analysis_points', [])
                if point_num <= len(analysis_points) and point_num > 0:
                    point_idx = analysis_points[point_num - 1]
                    loop_length = st['end'] - st['start']
                    
                    # Вычисляем размер окрестности
                    region_idx = region_num - 1
                    num_regions = 5
                    
                    fraction = 0.1 + (region_idx / (num_regions - 1)) * 0.9 if num_regions > 1 else 1.0
                    region_size = max(3, int(loop_length * fraction))
                    
                    half_size = region_size // 2
                    start = max(st['start'], point_idx - half_size)
                    end = min(st['end'], point_idx + half_size + 1)
                    
                    # Для последней окрестности берем всю петлю
                    if region_idx == num_regions - 1:
                        start = st['start']
                        end = st['end']
                    
                    if end > start:
                        found_regions.append({
                            'region_info': region_info,
                            'start': start,
                            'end': end,
                            'center': point_idx,
                            'x': vecg_x[start:end],
                            'y': vecg_y[start:end],
                            'z': vecg_z[start:end]
                        })
            else:
                # Total регион - весь ST
                found_regions.append({
                    'region_info': region_info,
                    'start': st['start'],
                    'end': st['end'],
                    'center': st['center'],
                    'x': vecg_x[st['start']:st['end']],
                    'y': vecg_y[st['start']:st['end']],
                    'z': vecg_z[st['start']:st['end']]
                })
    
    return found_regions


def plot_vecg_with_regions(vecg_x, vecg_y, vecg_z, found_regions, patient_idx, output_dir):
    """Визуализация ВЭКГ с выделением важных регионов в 3 проекциях"""
    
    # Создаем фигуру с 3 подграфиками для проекций
    fig = plt.figure(figsize=(18, 6))
    
    # Цвета для разных регионов
    colors = plt.cm.tab10(np.linspace(0, 1, len(found_regions)))
    
    # Проекция XY
    ax1 = fig.add_subplot(131)
    ax1.plot(vecg_x, vecg_y, 'k-', alpha=0.3, linewidth=0.5, label='ВЭКГ')
    for i, region in enumerate(found_regions):
        ax1.plot(region['x'], region['y'], 'o-', color=colors[i], 
                linewidth=2, markersize=4, 
                label=f"{region['region_info']['segment_type']} "
                      f"(imp: {region['region_info']['importance']:.4f})")
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Проекция XY', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.axis('equal')
    
    # Проекция XZ
    ax2 = fig.add_subplot(132)
    ax2.plot(vecg_x, vecg_z, 'k-', alpha=0.3, linewidth=0.5, label='ВЭКГ')
    for i, region in enumerate(found_regions):
        ax2.plot(region['x'], region['z'], 'o-', color=colors[i], 
                linewidth=2, markersize=4, 
                label=f"{region['region_info']['segment_type']} "
                      f"(imp: {region['region_info']['importance']:.4f})")
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Z', fontsize=12)
    ax2.set_title('Проекция XZ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axis('equal')
    
    # Проекция YZ
    ax3 = fig.add_subplot(133)
    ax3.plot(vecg_y, vecg_z, 'k-', alpha=0.3, linewidth=0.5, label='ВЭКГ')
    for i, region in enumerate(found_regions):
        ax3.plot(region['y'], region['z'], 'o-', color=colors[i], 
                linewidth=2, markersize=4, 
                label=f"{region['region_info']['segment_type']} "
                      f"(imp: {region['region_info']['importance']:.4f})")
    ax3.set_xlabel('Y', fontsize=12)
    ax3.set_ylabel('Z', fontsize=12)
    ax3.set_title('Проекция YZ', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.axis('equal')
    
    plt.suptitle(f'ВЭКГ пациента {patient_idx}: 5 наиболее важных участков', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Сохранение
    output_path = os.path.join(output_dir, f'vecg_patient_{patient_idx}_important_regions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен в {output_path}")
    plt.close()
    
    # Также создаем 3D визуализацию
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Вся ВЭКГ
    ax_3d.plot(vecg_x, vecg_y, vecg_z, 'k-', alpha=0.2, linewidth=0.5, label='ВЭКГ')
    
    # Важные регионы
    for i, region in enumerate(found_regions):
        ax_3d.plot(region['x'], region['y'], region['z'], 'o-', 
                  color=colors[i], linewidth=3, markersize=6,
                  label=f"{region['region_info']['segment_type']} "
                        f"(imp: {region['region_info']['importance']:.4f})")
    
    ax_3d.set_xlabel('X', fontsize=12)
    ax_3d.set_ylabel('Y', fontsize=12)
    ax_3d.set_zlabel('Z', fontsize=12)
    ax_3d.set_title(f'ВЭКГ пациента {patient_idx}: 3D визуализация важных участков', 
                   fontsize=14, fontweight='bold')
    ax_3d.legend(fontsize=9)
    
    output_path_3d = os.path.join(output_dir, f'vecg_patient_{patient_idx}_3d.png')
    plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
    print(f"3D график сохранен в {output_path_3d}")
    plt.close()


def plot_ecg_period_with_importance(avg_signal, important_regions, importances, feature_names,
                                    patient_idx, output_dir, sfreq=500, found_regions=None):
    """
    Построение одного периода ЭКГ с фоном, раскрашенным по важности всех признаков в розовых тонах
    
    Parameters:
    -----------
    avg_signal : np.ndarray
        Усредненный ЭКГ сигнал (один период)
    important_regions : list
        Список важных регионов (используется для локализации)
    importances : np.ndarray
        Массив важности всех признаков
    feature_names : list
        Список названий всех признаков
    patient_idx : int
        Номер пациента
    output_dir : str
        Директория для сохранения
    sfreq : float
        Частота дискретизации
    found_regions : list, optional
        Список найденных регионов в усредненном сигнале
    """
    signal_length = len(avg_signal)
    
    # Создаем вектор важности для каждой точки сигнала
    importance_vector = np.zeros(signal_length)
    
    print(f"\n  Построение графика периода ЭКГ с важностью всех признаков:")
    print(f"    Длина сигнала: {signal_length}")
    print(f"    Всего признаков: {len(feature_names)}")
    
    # Создаем словарь для быстрого доступа к важности по названию признака
    importance_dict = dict(zip(feature_names, importances))
    
    # Находим QRS и ST регионы в усредненном сигнале для локализации
    combined_signal = avg_signal.copy()
    try:
        qrs_regions = detect_qrs_regions(combined_signal, fs=sfreq)
        st_regions = detect_st_regions(combined_signal, qrs_regions, fs=sfreq)
    except:
        qrs_regions = []
        st_regions = []
    
    # Создаем словарь регионов по типу
    regions_dict = {}
    if qrs_regions:
        regions_dict['QRS'] = qrs_regions[0] if len(qrs_regions) > 0 else None
    if st_regions:
        regions_dict['ST'] = st_regions[0] if len(st_regions) > 0 else None
    
    # Обрабатываем все признаки и распределяем их важность по сигналу
    print(f"    Распределение важности всех признаков по сигналу...")
    
    for feature_name, importance in importance_dict.items():
        if importance <= 0:
            continue
        
        # Парсим название признака
        parts = feature_name.split('_')
        if len(parts) < 2:
            continue
        
        segment_type = parts[0]  # QRS или ST
        
        # Определяем границы региона
        if segment_type in regions_dict and regions_dict[segment_type] is not None:
            base_region = regions_dict[segment_type]
            
            # Проверяем, есть ли информация о точке и регионе
            if len(parts) >= 5 and parts[3].isdigit():
                point_num = int(parts[3])
                region_num = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None
                
                if point_num is not None and region_num is not None:
                    # Используем analysis_points из базового региона
                    analysis_points = base_region.get('analysis_points', [])
                    if point_num <= len(analysis_points) and point_num > 0:
                        point_idx = analysis_points[point_num - 1]
                        region_length = base_region['end'] - base_region['start']
                        
                        # Вычисляем размер окрестности с расширением для лучшего покрытия
                        region_idx = region_num - 1
                        num_regions = 5
                        
                        # Для расширяющихся окрестностей используем более широкие границы
                        # Чтобы покрыть весь комплекс, а не только маленькие участки
                        if region_idx == num_regions - 1:
                            # Последний регион - весь комплекс
                            start_idx = base_region['start']
                            end_idx = base_region['end']
                        else:
                            # Для других регионов используем расширенные окрестности
                            # Увеличиваем fraction для лучшего покрытия
                            fraction = 0.3 + (region_idx / (num_regions - 1)) * 0.7 if num_regions > 1 else 1.0
                            region_size = max(region_length // 2, int(region_length * fraction))
                            
                            half_size = region_size // 2
                            start_idx = max(base_region['start'], point_idx - half_size)
                            end_idx = min(base_region['end'], point_idx + half_size)
                            
                            # Значительно расширяем границы для лучшего покрытия всего комплекса
                            expansion = max(region_length // 4, 10)
                            start_idx = max(base_region['start'], start_idx - expansion)
                            end_idx = min(base_region['end'], end_idx + expansion)
                            
                            # Убеждаемся, что мы покрываем значительную часть комплекса
                            min_coverage = region_length // 2
                            if (end_idx - start_idx) < min_coverage:
                                center = (start_idx + end_idx) // 2
                                start_idx = max(base_region['start'], center - min_coverage // 2)
                                end_idx = min(base_region['end'], center + min_coverage // 2)
                    else:
                        # Fallback: используем весь базовый регион
                        start_idx = base_region['start']
                        end_idx = base_region['end']
                else:
                    # Total регион - весь базовый регион
                    start_idx = base_region['start']
                    end_idx = base_region['end']
            else:
                # Total регион - весь базовый регион
                start_idx = base_region['start']
                end_idx = base_region['end']
            
            # Убеждаемся, что индексы валидны
            start_idx = max(0, min(start_idx, signal_length - 1))
            end_idx = max(start_idx + 1, min(end_idx, signal_length))
            
            # Добавляем важность к вектору с плавным распределением
            if end_idx > start_idx:
                # Используем плавное распределение для лучшей визуализации
                region_size = end_idx - start_idx
                if region_size > 1:
                    # Создаем плавное распределение важности внутри региона
                    # Используем более равномерное распределение для лучшего покрытия
                    weights = np.ones(region_size)
                    # Делаем края немного менее важными для плавности, но не слишком сильно
                    if region_size > 20:
                        fade_size = min(region_size // 8, 10)
                        fade = np.linspace(0.8, 1.0, fade_size)
                        weights[:fade_size] = fade
                        weights[-fade_size:] = fade[::-1]
                    
                    importance_vector[start_idx:end_idx] += importance * weights
                else:
                    importance_vector[start_idx:end_idx] += importance
                
                # Также добавляем небольшую важность в соседние области для плавности
                if start_idx > 0:
                    fade_size = min(5, start_idx)
                    fade_weights = np.linspace(0.1, 0.3, fade_size)
                    importance_vector[start_idx - fade_size:start_idx] += importance * fade_weights
                if end_idx < signal_length:
                    fade_size = min(5, signal_length - end_idx)
                    fade_weights = np.linspace(0.3, 0.1, fade_size)
                    importance_vector[end_idx:end_idx + fade_size] += importance * fade_weights
        else:
            # Если регион не найден, используем пропорциональное распределение
            if segment_type == 'QRS':
                start_frac = 0.0
                end_frac = 0.3
            elif segment_type == 'ST':
                start_frac = 0.3
                end_frac = 0.6
            else:
                # Распределяем равномерно
                start_frac = 0.0
                end_frac = 1.0
            
            start_idx = int(start_frac * signal_length)
            end_idx = int(end_frac * signal_length)
            start_idx = max(0, min(start_idx, signal_length - 1))
            end_idx = max(start_idx + 1, min(end_idx, signal_length))
            
            if end_idx > start_idx:
                importance_vector[start_idx:end_idx] += importance
    
    # Добавляем базовое покрытие для QRS и ST комплексов
    # чтобы гарантировать, что все участки имеют цвет
    if 'QRS' in regions_dict and regions_dict['QRS'] is not None:
        qrs_region = regions_dict['QRS']
        qrs_start = qrs_region['start']
        qrs_end = qrs_region['end']
        # Добавляем минимальную важность для всего QRS комплекса
        qrs_importance = importance_vector[qrs_start:qrs_end].max()
        if qrs_importance < 0.1:  # Если важность слишком низкая
            # Добавляем базовую важность для всего QRS
            base_qrs_importance = 0.1
            importance_vector[qrs_start:qrs_end] = np.maximum(
                importance_vector[qrs_start:qrs_end],
                base_qrs_importance
            )
    
    if 'ST' in regions_dict and regions_dict['ST'] is not None:
        st_region = regions_dict['ST']
        st_start = st_region['start']
        st_end = st_region['end']
        # Добавляем минимальную важность для всего ST комплекса
        st_importance = importance_vector[st_start:st_end].max()
        if st_importance < 0.1:  # Если важность слишком низкая
            # Добавляем базовую важность для всего ST
            base_st_importance = 0.1
            importance_vector[st_start:st_end] = np.maximum(
                importance_vector[st_start:st_end],
                base_st_importance
            )
    
    # Нормализуем вектор важности
    if importance_vector.max() > 0:
        importance_vector = importance_vector / importance_vector.max()
    
    # Создаем розовую цветовую схему
    colors_pink = ['#FFFFFF', '#FFE4E1', '#FFB6C1', '#FF91A4', '#FF69B4', '#FF1493', '#DC143C']
    from matplotlib.colors import LinearSegmentedColormap
    pink_cmap = LinearSegmentedColormap.from_list('pink', colors_pink, N=256)
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Временная ось
    time_axis = np.arange(signal_length) / sfreq
    
    # Нормализуем сигнал для визуализации
    if avg_signal.std() > 0:
        signal_norm = (avg_signal - avg_signal.mean()) / avg_signal.std()
    else:
        signal_norm = avg_signal
    
    # Создаем фон с цветом важности
    # Используем fill_between для создания цветного фона
    y_min = signal_norm.min() - 0.5
    y_max = signal_norm.max() + 0.5
    
    # Создаем сетку для плавного градиента
    num_segments = signal_length
    for i in range(num_segments - 1):
        importance_val = (importance_vector[i] + importance_vector[i+1]) / 2
        color = pink_cmap(importance_val)
        
        # Рисуем прямоугольник для этого сегмента
        ax.fill_between([time_axis[i], time_axis[i+1]], 
                        [y_min, y_min], [y_max, y_max],
                        color=color, alpha=0.6, linewidth=0)
    
    # Рисуем сам сигнал ЭКГ поверх фона
    ax.plot(time_axis, signal_norm, 'k-', linewidth=2, label='ЭКГ сигнал', zorder=10)
    
    # Добавляем подписи топ-5 наиболее важных регионов
    if important_regions and len(important_regions) > 0:
        top_regions = sorted(important_regions, key=lambda x: x['importance'], reverse=True)[:5]
        for i, region_info in enumerate(top_regions):
            # Находим соответствующий регион в сигнале
            segment_type = region_info['segment_type']
            if segment_type in regions_dict and regions_dict[segment_type] is not None:
                base_region = regions_dict[segment_type]
                center_idx = (base_region['start'] + base_region['end']) // 2
                if center_idx < signal_length:
                    center_time = time_axis[center_idx]
                    center_y = signal_norm[center_idx] if center_idx < len(signal_norm) else 0
                    ax.annotate(f"{region_info['feature_name']}\n({region_info['importance']:.4f})",
                              xy=(center_time, center_y),
                              xytext=(center_time, center_y + 0.3),
                              fontsize=8,
                              ha='center',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                              arrowprops=dict(arrowstyle='->', lw=1),
                              zorder=20)
    
    ax.set_xlabel('Время (секунды)', fontsize=12)
    ax.set_ylabel('Амплитуда (нормализованная)', fontsize=12)
    ax.set_title(f'Усредненный период ЭКГ пациента {patient_idx} с важностью участков', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Добавляем цветовую шкалу
    sm = plt.cm.ScalarMappable(cmap=pink_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Важность признака', fontsize=11)
    
    plt.tight_layout()
    
    # Сохранение
    output_path = os.path.join(output_dir, f'ecg_period_importance_patient_{patient_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График периода ЭКГ с важностью сохранен в {output_path}")
    plt.close()


def plot_ecg_importance_heatmap(ecg_leads, important_regions, importances, feature_names, 
                                  patient_idx, output_dir, sfreq=500, found_regions=None):
    """
    Построение heatmap важности признаков для исходного ЭКГ сигнала в розовых тонах
    
    Parameters:
    -----------
    ecg_leads : dict
        Словарь с отведениями ЭКГ {'V1': ..., 'V2': ..., ...}
    important_regions : list
        Список важных регионов
    importances : np.ndarray
        Массив важности признаков
    feature_names : list
        Список названий признаков
    patient_idx : int
        Номер пациента
    output_dir : str
        Директория для сохранения
    sfreq : float
        Частота дискретизации
    found_regions : list, optional
        Список найденных регионов в усредненном сигнале для более точной локализации
    """
    # Получаем названия отведений
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Создаем матрицу важности для каждого отведения и временной точки
    # Используем длину самого длинного отведения
    max_length = max(len(ecg_leads.get(name, [])) for name in lead_names if name in ecg_leads)
    
    print(f"\n  Параметры heatmap:")
    print(f"    Количество отведений: {len(lead_names)}")
    print(f"    Максимальная длина сигнала: {max_length}")
    print(f"    Количество важных регионов: {len(important_regions)}")
    print(f"    Найдено регионов для локализации: {len(found_regions) if found_regions else 0}")
    
    # Создаем матрицу важности [отведения x время]
    importance_matrix = np.zeros((len(lead_names), max_length))
    
    # Находим QRS и ST регионы напрямую в исходном сигнале
    # Используем комбинацию всех отведений для поиска
    combined_signal = np.zeros(max_length)
    for lead_name in lead_names:
        if lead_name in ecg_leads:
            signal = ecg_leads[lead_name]
            if len(signal) > max_length:
                signal = signal[:max_length]
            elif len(signal) < max_length:
                signal = np.pad(signal, (0, max_length - len(signal)), mode='constant')
            combined_signal += np.abs(signal)
    
    # Нормализуем комбинированный сигнал
    if combined_signal.max() > 0:
        combined_signal = combined_signal / combined_signal.max()
    
    # Находим QRS и ST регионы в исходном сигнале
    try:
        qrs_regions_raw = detect_qrs_regions(combined_signal, fs=sfreq)
        st_regions_raw = detect_st_regions(combined_signal, qrs_regions_raw, fs=sfreq)
        print(f"  Найдено QRS регионов в исходном сигнале: {len(qrs_regions_raw)}")
        print(f"  Найдено ST регионов в исходном сигнале: {len(st_regions_raw)}")
    except Exception as e:
        print(f"  Ошибка при поиске регионов в исходном сигнале: {e}")
        qrs_regions_raw = []
        st_regions_raw = []
    
    # Используем регионы из исходного сигнала для локализации важных участков
    if (qrs_regions_raw or st_regions_raw) and len(important_regions) > 0:
        print(f"  Используем регионы из исходного сигнала для локализации")
        
        # Создаем словарь регионов по типу сегмента
        regions_dict = {}
        if qrs_regions_raw:
            regions_dict['QRS'] = qrs_regions_raw[0] if len(qrs_regions_raw) > 0 else None
        if st_regions_raw:
            regions_dict['ST'] = st_regions_raw[0] if len(st_regions_raw) > 0 else None
        
        for i, region_info in enumerate(important_regions):
            segment_type = region_info['segment_type']
            importance = region_info['importance']
            point_num = region_info['point_num']
            region_num = region_info['region_num']
            
            if segment_type in regions_dict and regions_dict[segment_type] is not None:
                base_region = regions_dict[segment_type]
                
                # Определяем границы региона
                if point_num is not None and region_num is not None:
                    # Используем analysis_points из базового региона
                    analysis_points = base_region.get('analysis_points', [])
                    if point_num <= len(analysis_points) and point_num > 0:
                        point_idx = analysis_points[point_num - 1]
                        region_length = base_region['end'] - base_region['start']
                        
                        # Вычисляем размер окрестности
                        region_idx = region_num - 1
                        num_regions = 5
                        fraction = 0.1 + (region_idx / (num_regions - 1)) * 0.9 if num_regions > 1 else 1.0
                        region_size = max(10, int(region_length * fraction))
                        
                        half_size = region_size // 2
                        start_idx = max(base_region['start'], point_idx - half_size)
                        end_idx = min(base_region['end'], point_idx + half_size)
                        
                        if region_idx == num_regions - 1:
                            start_idx = base_region['start']
                            end_idx = base_region['end']
                    else:
                        # Fallback: используем весь базовый регион
                        start_idx = base_region['start']
                        end_idx = base_region['end']
                else:
                    # Total регион - весь базовый регион
                    start_idx = base_region['start']
                    end_idx = base_region['end']
                
                # Убеждаемся, что индексы валидны
                start_idx = max(0, min(start_idx, max_length - 1))
                end_idx = max(start_idx + 10, min(end_idx, max_length))
                
                print(f"    Регион {i+1}: {region_info['feature_name']}, важность: {importance:.6f}")
                print(f"      Тип: {segment_type}, индексы: [{start_idx}, {end_idx}], размер: {end_idx - start_idx}")
                
                # Распределяем важность по всем отведениям
                if end_idx > start_idx:
                    for lead_idx in range(len(lead_names)):
                        importance_matrix[lead_idx, start_idx:end_idx] += importance
                    print(f"      ✓ Важность добавлена в матрицу")
            else:
                print(f"    Регион {i+1}: {region_info['feature_name']} - тип {segment_type} не найден в исходном сигнале")
    
    # Если не удалось найти регионы в исходном сигнале, используем пропорциональное распределение
    elif found_regions and len(found_regions) > 0:
        # Используем реальные индексы из найденных регионов
        # Масштабируем индексы из усредненного сигнала к исходному
        avg_signal_length = len(found_regions[0]['x']) if found_regions else max_length
        
        print(f"  Используем найденные регионы для локализации")
        print(f"  Количество найденных регионов: {len(found_regions)}")
        print(f"  Количество важных регионов: {len(important_regions)}")
        print(f"  Длина усредненного сигнала: {avg_signal_length}, исходного: {max_length}")
        
        # Если усредненный сигнал слишком короткий, используем пропорциональное распределение
        if avg_signal_length < 50:  # Слишком короткий сигнал для точной локализации
            print(f"  ВНИМАНИЕ: Усредненный сигнал слишком короткий ({avg_signal_length}). Используем пропорциональное распределение.")
            
            # Используем пропорции из усредненного сигнала для распределения по исходному
            for i, region in enumerate(found_regions):
                if i < len(important_regions):
                    region_info = important_regions[i]
                    importance = region_info['importance']
                    
                    # Вычисляем пропорции в усредненном сигнале
                    start_frac = region['start'] / avg_signal_length if avg_signal_length > 0 else 0
                    end_frac = region['end'] / avg_signal_length if avg_signal_length > 0 else 1
                    
                    # Применяем пропорции к исходному сигналу
                    start_idx = int(start_frac * max_length)
                    end_idx = int(end_frac * max_length)
                    start_idx = max(0, min(start_idx, max_length - 1))
                    end_idx = max(start_idx + 10, min(end_idx, max_length))  # Минимум 10 отсчетов
                    
                    print(f"    Регион {i+1}: {region_info['feature_name']}, важность: {importance:.6f}")
                    print(f"      Пропорции в усредненном: [{start_frac:.3f}, {end_frac:.3f}]")
                    print(f"      Индексы в исходном: [{start_idx}, {end_idx}], размер: {end_idx - start_idx}")
                    
                    # Распределяем важность по всем отведениям
                    if end_idx > start_idx:
                        for lead_idx in range(len(lead_names)):
                            importance_matrix[lead_idx, start_idx:end_idx] += importance
                        print(f"      ✓ Важность добавлена в матрицу")
                    else:
                        print(f"      ✗ Пропущено: некорректные индексы")
        else:
            # Нормальное масштабирование для достаточно длинного сигнала
            scale_factor = max_length / avg_signal_length if avg_signal_length > 0 else 1.0
            print(f"  Коэффициент масштабирования: {scale_factor:.4f}")
            
            for i, region in enumerate(found_regions):
                if i < len(important_regions):
                    region_info = important_regions[i]
                    importance = region_info['importance']
                    
                    # Масштабируем индексы региона к исходному сигналу
                    start_idx = int(region['start'] * scale_factor)
                    end_idx = int(region['end'] * scale_factor)
                    start_idx = max(0, min(start_idx, max_length - 1))
                    end_idx = max(start_idx + 10, min(end_idx, max_length))  # Минимум 10 отсчетов
                    
                    print(f"    Регион {i+1}: {region_info['feature_name']}, важность: {importance:.6f}")
                    print(f"      Индексы в усредненном: [{region['start']}, {region['end']}]")
                    print(f"      Индексы в исходном: [{start_idx}, {end_idx}], размер: {end_idx - start_idx}")
                    
                    # Распределяем важность по всем отведениям
                    if end_idx > start_idx:
                        for lead_idx in range(len(lead_names)):
                            importance_matrix[lead_idx, start_idx:end_idx] += importance
                        print(f"      ✓ Важность добавлена в матрицу")
                    else:
                        print(f"      ✗ Пропущено: некорректные индексы")
    else:
        # Используем приблизительную локализацию
        print(f"  Используем приблизительную локализацию (найденные регионы отсутствуют)")
        print(f"  Количество важных регионов: {len(important_regions)}")
        for region_info in important_regions:
            feature_name = region_info['feature_name']
            importance = region_info['importance']
            
            # Определяем тип сегмента и его примерное расположение
            segment_type = region_info['segment_type']
            point_num = region_info['point_num']
            
            # Примерная локализация QRS и ST сегментов в сигнале
            if segment_type == 'QRS':
                # QRS комплекс обычно занимает первые 10-15% сигнала
                start_frac = 0.0
                end_frac = 0.15
                
                if point_num is not None:
                    # Распределяем важность по точкам внутри QRS
                    qrs_points = 5
                    point_start = start_frac + (point_num - 1) / qrs_points * (end_frac - start_frac)
                    point_end = start_frac + point_num / qrs_points * (end_frac - start_frac)
                else:
                    # Total - весь QRS
                    point_start = start_frac
                    point_end = end_frac
                    
            elif segment_type == 'ST':
                # ST сегмент следует после QRS
                start_frac = 0.15
                end_frac = 0.35
                
                if point_num is not None:
                    # Распределяем важность по точкам внутри ST
                    st_points = 5
                    point_start = start_frac + (point_num - 1) / st_points * (end_frac - start_frac)
                    point_end = start_frac + point_num / st_points * (end_frac - start_frac)
                else:
                    # Total - весь ST
                    point_start = start_frac
                    point_end = end_frac
            else:
                continue
            
            # Распределяем важность по временным точкам и отведениям
            start_idx = int(point_start * max_length)
            end_idx = int(point_end * max_length)
            start_idx = max(0, min(start_idx, max_length - 1))
            end_idx = max(start_idx + 1, min(end_idx, max_length))
            
            print(f"    Регион: {feature_name}, важность: {importance:.6f}")
            print(f"      Доли: [{point_start:.3f}, {point_end:.3f}], индексы: [{start_idx}, {end_idx}]")
            
            # Распределяем важность равномерно по всем отведениям
            if end_idx > start_idx:
                for lead_idx in range(len(lead_names)):
                    importance_matrix[lead_idx, start_idx:end_idx] += importance
                print(f"      ✓ Важность добавлена в матрицу")
            else:
                print(f"      ✗ Пропущено: некорректные индексы")
    
    # Проверяем, заполнена ли матрица
    max_importance = importance_matrix.max()
    print(f"\n  Проверка матрицы важности:")
    print(f"    Максимальное значение: {max_importance:.6f}")
    print(f"    Среднее значение: {importance_matrix.mean():.6f}")
    print(f"    Ненулевых элементов: {np.count_nonzero(importance_matrix)} из {importance_matrix.size}")
    
    if max_importance <= 0:
        print("  ВНИМАНИЕ: Матрица важности пустая! Используем равномерное распределение.")
        # Если матрица пустая, распределяем важность равномерно по всем регионам
        for i, region_info in enumerate(important_regions):
            importance = region_info['importance']
            # Распределяем по всему сигналу равномерно
            segment_size = max_length // len(important_regions) if len(important_regions) > 0 else max_length
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, max_length)
            if end_idx > start_idx:
                for lead_idx in range(len(lead_names)):
                    importance_matrix[lead_idx, start_idx:end_idx] = importance
                print(f"    Заполнен регион {i+1}: [{start_idx}, {end_idx}] с важностью {importance:.6f}")
        
        # Нормализуем после заполнения
        max_importance = importance_matrix.max()
        if max_importance > 0:
            print(f"    Матрица заполнена. Новый максимум перед нормализацией: {max_importance:.6f}")
            importance_matrix = importance_matrix / max_importance
            print(f"    Матрица нормализована. Новый максимум: {importance_matrix.max():.6f}")
    
    # Финальная нормализация (на случай, если матрица уже была заполнена)
    if max_importance > 0 and importance_matrix.max() != 1.0:
        importance_matrix = importance_matrix / importance_matrix.max()
    
    # Создаем фигуру с двумя подграфиками: heatmap и исходный сигнал
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 2])
    
    # Heatmap важности в розовых тонах
    # Создаем кастомную розовую палитру от белого до темно-розового
    colors_pink = ['#FFFFFF', '#FFE4E1', '#FFB6C1', '#FF91A4', '#FF69B4', '#FF1493', '#DC143C']
    from matplotlib.colors import LinearSegmentedColormap
    pink_cmap = LinearSegmentedColormap.from_list('pink', colors_pink, N=256)
    
    # Отладочная информация
    print(f"  Размер матрицы важности: {importance_matrix.shape}")
    print(f"  Максимальная важность: {importance_matrix.max():.6f}")
    print(f"  Средняя важность: {importance_matrix.mean():.6f}")
    print(f"  Количество ненулевых элементов: {np.count_nonzero(importance_matrix)}")
    
    # Создаем heatmap с явными границами значений
    vmin = 0.0
    vmax = max(importance_matrix.max(), 0.01)  # Минимум 0.01 для видимости
    im = ax1.imshow(importance_matrix, aspect='auto', cmap=pink_cmap, 
                    interpolation='bilinear', origin='lower', vmin=vmin, vmax=vmax)
    
    ax1.set_yticks(range(len(lead_names)))
    ax1.set_yticklabels(lead_names)
    ax1.set_xlabel('Время (отсчеты)', fontsize=12)
    ax1.set_ylabel('Отведения ЭКГ', fontsize=12)
    ax1.set_title(f'Heatmap важности признаков для пациента {patient_idx}', 
                  fontsize=14, fontweight='bold')
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Важность признака', fontsize=11)
    
    # Второй подграфик: исходный ЭКГ сигнал с наложенной важностью
    time_axis = np.arange(max_length) / sfreq  # В секундах
    
    # Нормализуем сигналы для визуализации
    for i, lead_name in enumerate(lead_names):
        if lead_name in ecg_leads:
            signal = ecg_leads[lead_name]
            if len(signal) > max_length:
                signal = signal[:max_length]
            elif len(signal) < max_length:
                signal = np.pad(signal, (0, max_length - len(signal)), mode='constant')
            
            # Нормализуем сигнал
            if signal.std() > 0:
                signal_norm = (signal - signal.mean()) / signal.std()
            else:
                signal_norm = signal
            
            # Смещаем сигналы для визуализации
            offset = i * 3
            signal_plot = signal_norm + offset
            
            # Рисуем базовую линию сигнала (серая)
            ax2.plot(time_axis, signal_plot, 'k-', alpha=0.3, linewidth=0.8, zorder=1)
            
            # Рисуем сигнал с цветом в зависимости от важности (более эффективно)
            # Используем scatter для более плавной визуализации
            importance_vals = importance_matrix[i, :]
            # Усиливаем видимость важности
            color_intensities = np.clip(importance_vals * 1.5, 0, 1)
            colors_for_points = pink_cmap(color_intensities)
            
            # Рисуем точки с цветом в зависимости от важности (каждая 10-я точка для производительности)
            step = max(1, max_length // 2000)  # Ограничиваем количество точек
            ax2.scatter(time_axis[::step], signal_plot[::step], 
                       c=colors_for_points[::step], s=15, alpha=0.7, zorder=2)
            
            # Добавляем подпись отведения
            ax2.text(time_axis[-1] * 1.02, offset, lead_name, 
                    fontsize=10, va='center', fontweight='bold')
    
    ax2.set_xlabel('Время (секунды)', fontsize=12)
    ax2.set_ylabel('Отведения ЭКГ (нормализованные)', fontsize=12)
    ax2.set_title('Исходный ЭКГ сигнал с наложенной важностью признаков', 
                  fontsize=14, fontweight='bold')
    ax2.set_yticks([i * 3 for i in range(len(lead_names))])
    ax2.set_yticklabels(lead_names)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    output_path = os.path.join(output_dir, f'ecg_importance_heatmap_patient_{patient_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap важности сохранен в {output_path}")
    plt.close()


def main():
    """Основная функция"""
    # Загрузка конфигурации
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)
    
    # Получаем номер пациента из конфигурации
    patient_idx = config.get('visualization', {}).get('patient_idx', 0)
    sfreq = config.get('sampling_frequency', 500)
    num_periods = config.get('processing', {}).get('num_periods', 5)
    
    print(f"\n{'='*60}")
    print(f"Визуализация ВЭКГ для пациента {patient_idx}")
    print(f"{'='*60}\n")
    
    # Загрузка данных пациента
    patient_data = load_patient_data(patient_idx, config)
    
    # Извлечение отведений
    V1, V2, V3, V4, V5, V6, I, II = extract_ecg_leads(patient_data)
    
    # Сохраняем исходные сигналы для heatmap
    V1_raw = V1.copy()
    V2_raw = V2.copy()
    V3_raw = V3.copy()
    V4_raw = V4.copy()
    V5_raw = V5.copy()
    V6_raw = V6.copy()
    I_raw = I.copy()
    II_raw = II.copy()
    
    # Предобработка
    print("Предобработка ЭКГ...")
    V1 = preprocess_ecg(V1, sfreq)
    V2 = preprocess_ecg(V2, sfreq)
    V3 = preprocess_ecg(V3, sfreq)
    V4 = preprocess_ecg(V4, sfreq)
    V5 = preprocess_ecg(V5, sfreq)
    V6 = preprocess_ecg(V6, sfreq)
    I = preprocess_ecg(I, sfreq)
    II = preprocess_ecg(II, sfreq)
    
    # Расчет ВЭКГ
    print("Расчет ВЭКГ...")
    vecg_x, vecg_y, vecg_z = calculate_vecg(V1, V2, V3, V4, V5, V6, I, II)
    
    # Усреднение по периодам
    print("Усреднение по периодам...")
    avg_x, avg_y, avg_z = average_periods(vecg_x, vecg_y, vecg_z, num_periods=num_periods, sfreq=sfreq)
    
    # Загрузка feature importance из файла (избегаем проблем с версиями scikit-learn)
    print("Загрузка feature importance из Random Forest...")
    model_dir = os.path.join(script_dir, "models", "random_forest")
    feature_importance_path = os.path.join(model_dir, "feature_importance.csv")
    
    # Загрузка признаков для получения названий
    features_dir = os.path.join(script_dir, "features")
    train_features_path = os.path.join(features_dir, "train_features.csv")
    
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"Файл признаков не найден: {train_features_path}")
    
    feature_df = pd.read_csv(train_features_path)
    feature_names = feature_df.columns.tolist()
    
    # Пытаемся загрузить feature importance из CSV файла
    if os.path.exists(feature_importance_path):
        print(f"Загрузка feature importance из {feature_importance_path}...")
        importance_df = pd.read_csv(feature_importance_path)
        # Создаем словарь для быстрого доступа
        importance_dict = dict(zip(importance_df['feature_name'], importance_df['importance']))
        # Создаем массив важности в том же порядке, что и feature_names
        importances = np.array([importance_dict.get(name, 0.0) for name in feature_names])
    else:
        # Если файл не найден, пытаемся загрузить модель
        print("Файл feature_importance.csv не найден, пытаемся загрузить модель...")
        model_path = os.path.join(model_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найдены ни feature_importance.csv, ни model.pkl.\n"
                f"Пожалуйста, переобучите модель, запустив:\n"
                f"python classification/train_random_forest.py"
            )
        
        try:
            model = joblib.load(model_path)
            from classification.utils import get_feature_importance
            importances = get_feature_importance(model, feature_names)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Попробуйте переобучить модель с текущей версией scikit-learn:")
            print("python classification/train_random_forest.py")
            raise
    
    # Получение 5 наиболее важных регионов
    print("Определение наиболее важных регионов...")
    important_regions = get_important_regions_from_importances(importances, feature_names, top_n=5)
    
    print("\n5 наиболее важных регионов:")
    for i, region in enumerate(important_regions, 1):
        print(f"{i}. {region['feature_name']}: {region['importance']:.6f}")
    
    # Поиск регионов в ВЭКГ
    print("\nПоиск регионов в ВЭКГ...")
    found_regions = find_regions_in_vecg(avg_x, avg_y, avg_z, important_regions, sfreq=sfreq)
    
    print(f"Найдено регионов: {len(found_regions)}")
    
    if len(found_regions) == 0:
        print("Предупреждение: не найдено ни одного региона для визуализации!")
        print("Попробуйте выбрать другого пациента или проверьте данные.")
        return
    
    # Создание директории для результатов
    output_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Визуализация
    print("\nСоздание визуализаций...")
    plot_vecg_with_regions(avg_x, avg_y, avg_z, found_regions, patient_idx, output_dir)
    
    # Построение графика одного периода ЭКГ с важностью участков
    print("\nПостроение графика периода ЭКГ с важностью участков...")
    # Используем комбинацию всех компонент ВЭКГ для визуализации
    combined_avg = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
    plot_ecg_period_with_importance(combined_avg, important_regions, importances, feature_names,
                                    patient_idx, output_dir, sfreq=sfreq, found_regions=found_regions)
    
    print("\nГотово!")


if __name__ == "__main__":
    main()
