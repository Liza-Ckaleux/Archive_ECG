"""
Скрипт для извлечения feature importance из модели Random Forest
и сохранения в CSV файл для использования в визуализации
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classification.data_loader import load_data, get_base_dir

def extract_feature_importance():
    """Извлечение feature importance из модели или переобучение"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(get_base_dir(), "random_forest")
    model_path = os.path.join(model_dir, "model.pkl")
    feature_importance_path = os.path.join(model_dir, "feature_importance.csv")
    
    # Загрузка данных для получения названий признаков
    print("Загрузка данных для получения названий признаков...")
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # Пытаемся загрузить модель
    if os.path.exists(model_path):
        print(f"Попытка загрузки модели из {model_path}...")
        try:
            model = joblib.load(model_path)
            print("Модель успешно загружена!")
            
            # Извлекаем feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                print("Модель не имеет атрибута feature_importances_")
                raise ValueError("Модель не поддерживает feature importance")
            
            # Сохраняем в CSV
            feature_importance_df = pd.DataFrame({
                'feature_name': feature_names,
                'importance': importances
            })
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            feature_importance_df.to_csv(feature_importance_path, index=False)
            
            print(f"Feature importance сохранен в {feature_importance_path}")
            print(f"\nТоп-10 наиболее важных признаков:")
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"  {row['feature_name']}: {row['importance']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("\nПроблема с совместимостью версий scikit-learn.")
            print("Переобучение модели с текущей версией...")
            
            # Переобучение модели
            from classification.train_random_forest import train_random_forest
            
            base_dir = get_base_dir()
            result = train_random_forest(X_train, X_test, y_train, y_test, base_dir)
            
            print("\nМодель переобучена! Feature importance должен быть сохранен автоматически.")
            return True
    else:
        print(f"Модель не найдена: {model_path}")
        print("Переобучение модели...")
        
        # Переобучение модели
        from classification.train_random_forest import train_random_forest
        
        base_dir = get_base_dir()
        result = train_random_forest(X_train, X_test, y_train, y_test, base_dir)
        
        print("\nМодель обучена! Feature importance должен быть сохранен автоматически.")
        return True


if __name__ == "__main__":
    print("="*60)
    print("Извлечение feature importance из Random Forest")
    print("="*60)
    extract_feature_importance()
    print("\nГотово!")
