"""
Общая функция для загрузки данных
"""
import os
import pandas as pd
import numpy as np
import pickle
import yaml


def load_config(config_path=None):
    """Загрузка конфигурации из YAML файла"""
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(config_path):
        return {'dataset_type': 'original'}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_data():
    """Загрузка признаков и меток"""
    # Определяем корневую директорию проекта (на уровень выше classification/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Загрузка данных...")
    
    # Загружаем конфигурацию для определения типа датасета
    config = load_config()
    dataset_type = config.get('dataset_type', 'original')
    
    # Загружаем признаки из папки features
    features_dir = os.path.join(project_root, "features")
    train_features_path = os.path.join(features_dir, "train_features.csv")
    test_features_path = os.path.join(features_dir, "test_features.csv")
    
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"Файл {train_features_path} не найден. Сначала запустите process_ecg.py")
    if not os.path.exists(test_features_path):
        raise FileNotFoundError(f"Файл {test_features_path} не найден. Сначала запустите process_ecg.py")
    
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    
    # Загружаем метки в зависимости от типа датасета
    dataset_dir = config.get('dataset_dir')
    
    if dataset_type == "stratified":
        y_train_path = os.path.join(dataset_dir, "y_train_stratified.pkl")
        y_test_path = os.path.join(dataset_dir, "y_test_stratified.pkl")
        print(f"Используются стратифицированные метки")
    else:
        y_train_path = os.path.join(dataset_dir, "y_train.pkl")
        y_test_path = os.path.join(dataset_dir, "y_test.pkl")
        print(f"Используются исходные метки")
    
    # Проверяем существование файлов
    if not os.path.exists(y_train_path):
        print(f"Предупреждение: файл {y_train_path} не найден, пробуем исходные метки")
        y_train_path = os.path.join(dataset_dir, "y_train.pkl")
        y_test_path = os.path.join(dataset_dir, "y_test.pkl")
    
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"Файл меток {y_train_path} не найден!")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Файл меток {y_test_path} не найден!")
    
    with open(y_train_path, 'rb') as f:
        y_train_raw = pickle.load(f)
    
    with open(y_test_path, 'rb') as f:
        y_test_raw = pickle.load(f)
    
    # Извлекаем метки из первой строки
    def extract_labels(y_data):
        if isinstance(y_data, np.ndarray):
            if y_data.ndim == 1:
                return y_data.astype(float)
            elif y_data.ndim == 2:
                return y_data[0, :].astype(float)
            else:
                return y_data[0].flatten().astype(float)
        elif isinstance(y_data, (list, tuple)):
            if len(y_data) == 0:
                return np.array([])
            first_item = y_data[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                return np.array(first_item).flatten().astype(float)
            else:
                return np.array([float(item) for item in y_data])
        else:
            try:
                return np.array([float(y_data)]).flatten()
            except:
                return np.array([0.0])
    
    y_train = extract_labels(y_train_raw)
    y_test = extract_labels(y_test_raw)
    
    print(f"\nМетки до обработки:")
    print(f"  Train: форма={y_train.shape}, уникальные={np.unique(y_train)}, распределение={np.bincount(y_train.astype(int)) if len(np.unique(y_train.astype(int))) <= 2 else 'много классов'}")
    print(f"  Test: форма={y_test.shape}, уникальные={np.unique(y_test)}, распределение={np.bincount(y_test.astype(int)) if len(np.unique(y_test.astype(int))) <= 2 else 'много классов'}")
    
    # Преобразуем в бинарные метки (0 и 1) и в int
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    if len(unique_train) > 2:
        threshold = np.median(unique_train)
        y_train = (y_train > threshold).astype(int)
    else:
        y_min = y_train.min()
        y_max = y_train.max()
        if y_min != 0 or y_max != 1:
            if y_max > y_min:
                y_train = ((y_train - y_min) / (y_max - y_min)).round().astype(int)
            else:
                y_train = np.zeros_like(y_train, dtype=int)
        else:
            y_train = y_train.astype(int)
    
    if len(unique_test) > 2:
        threshold = np.median(unique_test)
        y_test = (y_test > threshold).astype(int)
    else:
        y_min = y_test.min()
        y_max = y_test.max()
        if y_min != 0 or y_max != 1:
            if y_max > y_min:
                y_test = ((y_test - y_min) / (y_max - y_min)).round().astype(int)
            else:
                y_test = np.zeros_like(y_test, dtype=int)
        else:
            y_test = y_test.astype(int)
    
    print(f"\nМетки после обработки:")
    print(f"  Train: форма={y_train.shape}, уникальные={np.unique(y_train)}, распределение={np.bincount(y_train)}")
    print(f"  Test: форма={y_test.shape}, уникальные={np.unique(y_test)}, распределение={np.bincount(y_test)}")
    
    # Проверяем, что количество образцов совпадает
    if len(X_train) != len(y_train):
        print(f"Предупреждение: несоответствие размеров train - признаки: {len(X_train)}, метки: {len(y_train)}")
        min_len = min(len(X_train), len(y_train))
        X_train = X_train.iloc[:min_len]
        y_train = y_train[:min_len]
        print(f"  Обрезано до {min_len} образцов")
    
    if len(X_test) != len(y_test):
        print(f"Предупреждение: несоответствие размеров test - признаки: {len(X_test)}, метки: {len(y_test)}")
        min_len = min(len(X_test), len(y_test))
        X_test = X_test.iloc[:min_len]
        y_test = y_test[:min_len]
        print(f"  Обрезано до {min_len} образцов")
    
    print(f"\nФинальные размеры:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, распределение классов: {np.bincount(y_train)}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}, распределение классов: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


def get_base_dir():
    """Получить базовую директорию для сохранения результатов"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "models")

