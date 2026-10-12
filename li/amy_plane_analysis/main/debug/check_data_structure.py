"""
Скрипт для проверки структуры данных
"""
import pickle
import numpy as np

def check_data_structure(filepath):
    """Проверка структуры данных в pickle файле"""
    print(f"\nПроверка файла: {filepath}")
    print("=" * 60)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Тип данных: {type(data)}")
    
    if isinstance(data, np.ndarray):
        print(f"Форма массива: {data.shape}")
        print(f"Тип данных: {data.dtype}")
        print(f"Размерность: {data.ndim}")
        if data.ndim >= 1:
            print(f"Первые значения: {data.flat[:10]}")
    elif isinstance(data, (list, tuple)):
        print(f"Длина: {len(data)}")
        if len(data) > 0:
            print(f"Тип первого элемента: {type(data[0])}")
            if isinstance(data[0], np.ndarray):
                print(f"Форма первого элемента: {data[0].shape}")
    elif isinstance(data, dict):
        print(f"Ключи: {list(data.keys())[:10]}")
        if len(data) > 0:
            first_key = list(data.keys())[0]
            print(f"Тип значения для ключа '{first_key}': {type(data[first_key])}")
            if isinstance(data[first_key], np.ndarray):
                print(f"Форма значения: {data[first_key].shape}")
    
    return data

if __name__ == "__main__":
    train_path = "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset/X_train.pkl"
    test_path = "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset/X_test.pkl"
    
    train_data = check_data_structure(train_path)
    test_data = check_data_structure(test_path)

