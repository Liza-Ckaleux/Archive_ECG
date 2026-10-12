"""
Скрипт для стратифицированного разделения данных на train/test
Сохраняет распределение классов в обеих выборках
"""
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os


def extract_labels(y_data):
    """Извлечение меток из разных форматов"""
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


def normalize_labels(y):
    """Нормализация меток к бинарным (0 и 1)"""
    unique = np.unique(y)
    
    if len(unique) > 2:
        threshold = np.median(unique)
        y = (y > threshold).astype(int)
    else:
        y_min = y.min()
        y_max = y.max()
        if y_min != 0 or y_max != 1:
            if y_max > y_min:
                y = ((y - y_min) / (y_max - y_min)).round().astype(int)
            else:
                y = np.zeros_like(y, dtype=int)
        else:
            y = y.astype(int)
    
    return y


def split_dataset_stratified(dataset_dir, test_size=0.2, random_state=42):
    """
    Стратифицированное разделение данных на train/test
    
    Parameters:
    -----------
    dataset_dir : str
        Директория с исходными данными
    test_size : float
        Доля тестовой выборки (по умолчанию 0.2)
    random_state : int
        Seed для воспроизводимости
    """
    print("Загрузка исходных данных...")
    
    # Загружаем train данные
    X_train_path = os.path.join(dataset_dir, "X_train.pkl")
    y_train_path = os.path.join(dataset_dir, "y_train.pkl")
    
    with open(X_train_path, 'rb') as f:
        X_train_old = pickle.load(f)
    
    with open(y_train_path, 'rb') as f:
        y_train_old_raw = pickle.load(f)
    
    y_train_old = extract_labels(y_train_old_raw)
    y_train_old = normalize_labels(y_train_old)
    
    # Загружаем test данные
    X_test_path = os.path.join(dataset_dir, "X_test.pkl")
    y_test_path = os.path.join(dataset_dir, "y_test.pkl")
    
    with open(X_test_path, 'rb') as f:
        X_test_old = pickle.load(f)
    
    with open(y_test_path, 'rb') as f:
        y_test_old_raw = pickle.load(f)
    
    y_test_old = extract_labels(y_test_old_raw)
    y_test_old = normalize_labels(y_test_old)
    
    print(f"Исходные данные:")
    print(f"  Train: X={X_train_old.shape if isinstance(X_train_old, np.ndarray) else 'unknown'}, "
          f"y={len(y_train_old)}, распределение классов: {np.bincount(y_train_old)}")
    print(f"  Test: X={X_test_old.shape if isinstance(X_test_old, np.ndarray) else 'unknown'}, "
          f"y={len(y_test_old)}, распределение классов: {np.bincount(y_test_old)}")
    
    # Объединяем все данные
    if isinstance(X_train_old, np.ndarray) and isinstance(X_test_old, np.ndarray):
        X_all = np.concatenate([X_train_old, X_test_old], axis=0)
    elif isinstance(X_train_old, list) and isinstance(X_test_old, list):
        X_all = X_train_old + X_test_old
    else:
        # Если разные типы, преобразуем в списки
        X_train_list = X_train_old if isinstance(X_train_old, list) else [X_train_old[i] for i in range(len(y_train_old))]
        X_test_list = X_test_old if isinstance(X_test_old, list) else [X_test_old[i] for i in range(len(y_test_old))]
        X_all = X_train_list + X_test_list
    
    y_all = np.concatenate([y_train_old, y_test_old])
    
    print(f"\nОбъединенные данные:")
    print(f"  Всего образцов: {len(y_all)}")
    print(f"  Распределение классов: {np.bincount(y_all)}")
    print(f"  Процент класса 0: {np.sum(y_all == 0) / len(y_all) * 100:.2f}%")
    print(f"  Процент класса 1: {np.sum(y_all == 1) / len(y_all) * 100:.2f}%")
    
    # Стратифицированное разделение
    print(f"\nРазделение на train/test (test_size={test_size})...")
    
    if isinstance(X_all, np.ndarray):
        # Для numpy массивов
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_all, y_all, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_all
        )
    else:
        # Для списков используем индексы
        indices = np.arange(len(y_all))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y_all
        )
        
        X_train_new = [X_all[i] for i in train_indices]
        X_test_new = [X_all[i] for i in test_indices]
        y_train_new = y_all[train_indices]
        y_test_new = y_all[test_indices]
    
    print(f"\nНовые данные:")
    print(f"  Train: {len(y_train_new)} образцов, распределение: {np.bincount(y_train_new)}")
    print(f"  Test: {len(y_test_new)} образцов, распределение: {np.bincount(y_test_new)}")
    print(f"  Train - процент класса 0: {np.sum(y_train_new == 0) / len(y_train_new) * 100:.2f}%")
    print(f"  Train - процент класса 1: {np.sum(y_train_new == 1) / len(y_train_new) * 100:.2f}%")
    print(f"  Test - процент класса 0: {np.sum(y_test_new == 0) / len(y_test_new) * 100:.2f}%")
    print(f"  Test - процент класса 1: {np.sum(y_test_new == 1) / len(y_test_new) * 100:.2f}%")
    
    # Сохраняем метки в том же формате, что и исходные
    # Если исходные были в формате [2, N], сохраняем так же
    if isinstance(y_train_old_raw, np.ndarray) and y_train_old_raw.ndim == 2:
        # Сохраняем в формате [2, N] где первая строка - метки
        y_train_new_saved = np.zeros((2, len(y_train_new)), dtype=float)
        y_train_new_saved[0, :] = y_train_new
        y_train_new_saved[1, :] = y_train_new  # Дублируем для совместимости
        
        y_test_new_saved = np.zeros((2, len(y_test_new)), dtype=float)
        y_test_new_saved[0, :] = y_test_new
        y_test_new_saved[1, :] = y_test_new
    else:
        y_train_new_saved = y_train_new
        y_test_new_saved = y_test_new
    
    # Сохраняем новые файлы
    print(f"\nСохранение новых файлов в {dataset_dir}...")
    
    # Сохраняем с суффиксом _stratified
    X_train_new_path = os.path.join(dataset_dir, "X_train_stratified.pkl")
    y_train_new_path = os.path.join(dataset_dir, "y_train_stratified.pkl")
    X_test_new_path = os.path.join(dataset_dir, "X_test_stratified.pkl")
    y_test_new_path = os.path.join(dataset_dir, "y_test_stratified.pkl")
    
    with open(X_train_new_path, 'wb') as f:
        pickle.dump(X_train_new, f)
    
    with open(y_train_new_path, 'wb') as f:
        pickle.dump(y_train_new_saved, f)
    
    with open(X_test_new_path, 'wb') as f:
        pickle.dump(X_test_new, f)
    
    with open(y_test_new_path, 'wb') as f:
        pickle.dump(y_test_new_saved, f)
    
    print(f"✓ Сохранено:")
    print(f"  {X_train_new_path}")
    print(f"  {y_train_new_path}")
    print(f"  {X_test_new_path}")
    print(f"  {y_test_new_path}")
    
    return X_train_new, X_test_new, y_train_new, y_test_new


if __name__ == "__main__":
    dataset_dir = "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset"
    
    print("=" * 60)
    print("Стратифицированное разделение данных")
    print("=" * 60)
    
    split_dataset_stratified(dataset_dir, test_size=0.2, random_state=42)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)

