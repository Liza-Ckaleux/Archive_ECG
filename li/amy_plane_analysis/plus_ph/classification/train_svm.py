"""
Обучение модели SVM
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import evaluate_model, plot_roc_curve, plot_confusion_matrix
from data_loader import load_data, get_base_dir


def train_svm(X_train, X_test, y_train, y_test, base_dir):
    """Обучение и оценка SVM"""
    model_name = "SVM"
    model_dir = os.path.join(base_dir, "svm")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Обучение модели: {model_name}")
    print(f"{'='*60}")
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = SVC(probability=True, random_state=42, kernel='rbf', C=1, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Предсказания на тесте
    print(f"\nОценка на тестовой выборке:")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics, test_cm = evaluate_model(y_test, y_test_pred, y_test_proba,
                                           f"{model_name} (Test)")
    
    # Построение графиков
    plot_roc_curve(y_test, y_test_proba, f"{model_name}",
                  os.path.join(model_dir, "roc_curve.png"))
    
    plot_confusion_matrix(test_cm, f"{model_name}",
                         os.path.join(model_dir, "confusion_matrix.png"))
    
    # Сохранение модели и скейлера
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # Сохранение метрик
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(os.path.join(model_dir, "metrics.csv"), index=False)
    
    print(f"\nРезультаты сохранены в {model_dir}")
    
    return {
        'model': model,
        'scaler': scaler,
        'test_metrics': test_metrics,
        'test_cm': test_cm
    }


if __name__ == "__main__":
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data()
    
    # Базовая директория для результатов
    base_dir = get_base_dir()
    
    # Обучение модели
    result = train_svm(X_train, X_test, y_train, y_test, base_dir)
    
    print("\nГотово!")

