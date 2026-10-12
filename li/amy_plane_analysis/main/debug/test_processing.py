"""
Тестовый скрипт для проверки обработки на небольшом подмножестве данных
"""
import pickle
import numpy as np
import os
from process_ecg import process_patient, process_dataset

def test_single_patient():
    """Тест обработки одного пациента"""
    print("Тест обработки одного пациента...")
    
    # Загружаем данные
    train_path = "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset/X_train.pkl"
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Тип данных: {type(data)}")
    
    # Берем первого пациента
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            patient_data = data[0]
            print(f"Форма данных пациента: {patient_data.shape}")
        else:
            print(f"Неожиданная размерность: {data.ndim}")
            return
    elif isinstance(data, (list, tuple)):
        patient_data = data[0]
        print(f"Тип данных пациента: {type(patient_data)}")
    elif isinstance(data, dict):
        first_key = list(data.keys())[0]
        patient_data = data[first_key]
        print(f"Тип данных пациента: {type(patient_data)}")
    else:
        print("Неизвестный формат данных")
        return
    
    try:
        features = process_patient(patient_data, sfreq=500)
        print("\nУспешно обработан!")
        print(f"Количество признаков: {len(features)}")
        print(f"Признаки: {list(features.keys())}")
        print(f"\nЗначения признаков:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"\nОшибка при обработке: {e}")
        import traceback
        traceback.print_exc()

def test_small_subset():
    """Тест обработки небольшого подмножества"""
    print("\nТест обработки небольшого подмножества...")
    
    train_path = "/Users/moomintroll/Desktop/Научка_мага/Amy_dataset/X_train.pkl"
    
    # Загружаем данные
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    
    # Создаем подмножество из первых 5 пациентов
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            subset = data[:5]
        else:
            print("Неожиданная размерность")
            return
    elif isinstance(data, (list, tuple)):
        subset = data[:5]
    elif isinstance(data, dict):
        keys = list(data.keys())[:5]
        subset = {k: data[k] for k in keys}
    else:
        print("Неизвестный формат")
        return
    
    # Сохраняем подмножество
    test_path = "/Users/moomintroll/Desktop/Научка_мага/Amy_plane_analysis/test_subset.pkl"
    with open(test_path, 'wb') as f:
        pickle.dump(subset, f)
    
    # Обрабатываем подмножество
    try:
        # Создаем папку для признаков если её нет
        features_dir = "/Users/moomintroll/Desktop/Научка_мага/Amy_plane_analysis/features"
        os.makedirs(features_dir, exist_ok=True)
        
        result = process_dataset(test_path, 
                                 output_path=os.path.join(features_dir, "test_subset_features.csv"),
                                 sfreq=500)
        print(f"\nУспешно обработано {len(result)} пациентов")
        print(f"Форма результата: {result.shape}")
        print(f"\nПервые несколько строк:")
        print(result.head())
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_patient()
    test_small_subset()

