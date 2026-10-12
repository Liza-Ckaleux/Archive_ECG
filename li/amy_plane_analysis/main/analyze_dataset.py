"""
Скрипт для анализа датасета: размерность, соотношение классов, статистика
"""
import os
import sys
import numpy as np
import pickle
import yaml
import pandas as pd

def load_config(config_path=None):
    """Загрузка конфигурации из YAML файла"""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Файл конфигурации {config_path} не найден.")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_dataset_files(dataset_dir, dataset_type='original'):
    """Загрузка файлов датасета"""
    if dataset_type == "stratified":
        train_filename = "X_train_stratified.pkl"
        test_filename = "X_test_stratified.pkl"
        y_train_filename = "y_train_stratified.pkl"
        y_test_filename = "y_test_stratified.pkl"
    else:
        train_filename = "X_train.pkl"
        test_filename = "X_test.pkl"
        y_train_filename = "y_train.pkl"
        y_test_filename = "y_test.pkl"
    
    train_path = os.path.join(dataset_dir, train_filename)
    test_path = os.path.join(dataset_dir, test_filename)
    y_train_path = os.path.join(dataset_dir, y_train_filename)
    y_test_path = os.path.join(dataset_dir, y_test_filename)
    
    # Проверяем существование файлов
    files_exist = {
        'train': os.path.exists(train_path),
        'test': os.path.exists(test_path),
        'y_train': os.path.exists(y_train_path),
        'y_test': os.path.exists(y_test_path)
    }
    
    return {
        'train_path': train_path if files_exist['train'] else None,
        'test_path': test_path if files_exist['test'] else None,
        'y_train_path': y_train_path if files_exist['y_train'] else None,
        'y_test_path': y_test_path if files_exist['y_test'] else None,
        'files_exist': files_exist
    }


def extract_labels(y_data):
    """Извлечение меток из данных"""
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


def analyze_data_shape(data):
    """Анализ размерности данных"""
    if isinstance(data, np.ndarray):
        return {
            'type': 'numpy.ndarray',
            'shape': data.shape,
            'ndim': data.ndim,
            'dtype': str(data.dtype),
            'size': data.size
        }
    elif isinstance(data, (list, tuple)):
        return {
            'type': type(data).__name__,
            'length': len(data),
            'first_item_shape': np.array(data[0]).shape if len(data) > 0 else None
        }
    elif isinstance(data, dict):
        return {
            'type': 'dict',
            'keys': list(data.keys()),
            'num_keys': len(data.keys())
        }
    else:
        return {
            'type': type(data).__name__,
            'str': str(data)[:100]
        }


def analyze_labels(y_data, name=""):
    """Анализ меток"""
    y = extract_labels(y_data)
    
    if len(y) == 0:
        return {
            'name': name,
            'count': 0,
            'unique_values': [],
            'class_distribution': {},
            'percentage_ones': 0.0,
            'percentage_zeros': 0.0
        }
    
    unique_values = np.unique(y)
    
    # Преобразуем в бинарные метки для анализа
    if len(unique_values) > 2:
        threshold = np.median(unique_values)
        y_binary = (y > threshold).astype(int)
    else:
        y_min = y.min()
        y_max = y.max()
        if y_max > y_min:
            y_binary = ((y - y_min) / (y_max - y_min)).round().astype(int)
        else:
            y_binary = np.zeros_like(y, dtype=int)
    
    unique_binary = np.unique(y_binary)
    class_counts = {int(val): int(np.sum(y_binary == val)) for val in unique_binary}
    
    total = len(y_binary)
    percentage_ones = (class_counts.get(1, 0) / total * 100) if total > 0 else 0.0
    percentage_zeros = (class_counts.get(0, 0) / total * 100) if total > 0 else 0.0
    
    return {
        'name': name,
        'count': total,
        'unique_values': unique_values.tolist(),
        'class_distribution': class_counts,
        'percentage_ones': percentage_ones,
        'percentage_zeros': percentage_zeros,
        'ratio': class_counts.get(1, 0) / class_counts.get(0, 1) if class_counts.get(0, 0) > 0 else float('inf')
    }


def main():
    """Основная функция"""
    print("="*70)
    print("Анализ датасета")
    print("="*70)
    
    # Загрузка конфигурации
    config = load_config()
    if config is None:
        print("Не удалось загрузить конфигурацию!")
        return
    
    dataset_dir = config.get('dataset_dir')
    dataset_type = config.get('dataset_type', 'original')
    
    print(f"\nКонфигурация:")
    print(f"  Директория данных: {dataset_dir}")
    print(f"  Тип датасета: {dataset_type}")
    
    # Загрузка файлов
    files = load_dataset_files(dataset_dir, dataset_type)
    
    if not files['files_exist']['train']:
        print(f"\nОШИБКА: Файл {files['train_path']} не найден!")
        return
    
    if not files['files_exist']['test']:
        print(f"\nОШИБКА: Файл {files['test_path']} не найден!")
        return
    
    # Загрузка данных
    print(f"\nЗагрузка данных...")
    with open(files['train_path'], 'rb') as f:
        X_train_data = pickle.load(f)
    
    with open(files['test_path'], 'rb') as f:
        X_test_data = pickle.load(f)
    
    # Анализ размерности данных
    print(f"\n{'='*70}")
    print("РАЗМЕРНОСТЬ ДАННЫХ")
    print(f"{'='*70}")
    
    train_shape = analyze_data_shape(X_train_data)
    test_shape = analyze_data_shape(X_test_data)
    
    print(f"\nОбучающая выборка (X_train):")
    for key, value in train_shape.items():
        print(f"  {key}: {value}")
    
    print(f"\nТестовая выборка (X_test):")
    for key, value in test_shape.items():
        print(f"  {key}: {value}")
    
    # Определение количества образцов
    if isinstance(X_train_data, np.ndarray):
        if X_train_data.ndim == 3:
            n_train = X_train_data.shape[0]
            n_test = X_test_data.shape[0] if isinstance(X_test_data, np.ndarray) else 0
        else:
            n_train = len(X_train_data) if hasattr(X_train_data, '__len__') else 1
            n_test = len(X_test_data) if isinstance(X_test_data, np.ndarray) and hasattr(X_test_data, '__len__') else 1
    elif isinstance(X_train_data, (list, tuple)):
        n_train = len(X_train_data)
        n_test = len(X_test_data) if isinstance(X_test_data, (list, tuple)) else 0
    elif isinstance(X_train_data, dict):
        n_train = len(X_train_data)
        n_test = len(X_test_data) if isinstance(X_test_data, dict) else 0
    else:
        n_train = 1
        n_test = 1
    
    print(f"\nКоличество образцов:")
    print(f"  Train: {n_train}")
    print(f"  Test: {n_test}")
    print(f"  Всего: {n_train + n_test}")
    
    # Анализ меток
    print(f"\n{'='*70}")
    print("АНАЛИЗ МЕТОК")
    print(f"{'='*70}")
    
    if files['files_exist']['y_train'] and files['files_exist']['y_test']:
        print(f"\nЗагрузка меток...")
        with open(files['y_train_path'], 'rb') as f:
            y_train_raw = pickle.load(f)
        
        with open(files['y_test_path'], 'rb') as f:
            y_test_raw = pickle.load(f)
        
        train_labels = analyze_labels(y_train_raw, "Train")
        test_labels = analyze_labels(y_test_raw, "Test")
        
        print(f"\nОбучающая выборка (y_train):")
        print(f"  Количество образцов: {train_labels['count']}")
        print(f"  Уникальные значения: {train_labels['unique_values']}")
        print(f"  Распределение классов: {train_labels['class_distribution']}")
        print(f"  Процент единиц (класс 1): {train_labels['percentage_ones']:.2f}%")
        print(f"  Процент нулей (класс 0): {train_labels['percentage_zeros']:.2f}%")
        if train_labels['ratio'] != float('inf'):
            print(f"  Соотношение классов (1:0): 1:{train_labels['ratio']:.2f}")
        
        print(f"\nТестовая выборка (y_test):")
        print(f"  Количество образцов: {test_labels['count']}")
        print(f"  Уникальные значения: {test_labels['unique_values']}")
        print(f"  Распределение классов: {test_labels['class_distribution']}")
        print(f"  Процент единиц (класс 1): {test_labels['percentage_ones']:.2f}%")
        print(f"  Процент нулей (класс 0): {test_labels['percentage_zeros']:.2f}%")
        if test_labels['ratio'] != float('inf'):
            print(f"  Соотношение классов (1:0): 1:{test_labels['ratio']:.2f}")
        
        # Общая статистика
        total_samples = train_labels['count'] + test_labels['count']
        total_ones = train_labels['class_distribution'].get(1, 0) + test_labels['class_distribution'].get(1, 0)
        total_zeros = train_labels['class_distribution'].get(0, 0) + test_labels['class_distribution'].get(0, 0)
        total_percentage_ones = (total_ones / total_samples * 100) if total_samples > 0 else 0.0
        
        print(f"\nОбщая статистика:")
        print(f"  Всего образцов: {total_samples}")
        print(f"  Класс 0: {total_zeros} ({total_zeros/total_samples*100:.2f}%)")
        print(f"  Класс 1: {total_ones} ({total_percentage_ones:.2f}%)")
        if total_zeros > 0:
            print(f"  Соотношение классов (1:0): 1:{total_ones/total_zeros:.2f}")
        
        # Проверка соответствия размеров
        print(f"\nПроверка соответствия размеров:")
        if n_train == train_labels['count']:
            print(f"  ✓ Размеры X_train и y_train совпадают: {n_train}")
        else:
            print(f"  ✗ Несоответствие: X_train={n_train}, y_train={train_labels['count']}")
        
        if n_test == test_labels['count']:
            print(f"  ✓ Размеры X_test и y_test совпадают: {n_test}")
        else:
            print(f"  ✗ Несоответствие: X_test={n_test}, y_test={test_labels['count']}")
    else:
        print(f"\nМетки не найдены:")
        if not files['files_exist']['y_train']:
            print(f"  ✗ {files['y_train_path']}")
        if not files['files_exist']['y_test']:
            print(f"  ✗ {files['y_test_path']}")
    
    print(f"\n{'='*70}")
    print("Анализ завершен!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
