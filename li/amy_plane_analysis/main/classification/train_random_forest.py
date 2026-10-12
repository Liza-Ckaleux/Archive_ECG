"""
Обучение модели Random Forest
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    evaluate_model, plot_pr_curve, plot_metrics_vs_threshold,
    plot_confusion_matrix, plot_feature_importance,
    tune_threshold, predict_with_threshold, save_threshold,
)
from data_loader import load_data, get_base_dir


def train_random_forest(X_train, X_test, y_train, y_test, base_dir):
    """Обучение и оценка Random Forest"""
    model_name = "Random Forest"
    model_dir = os.path.join(base_dir, "random_forest")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Обучение модели: {model_name}")
    print(f"{'='*60}")
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
    model.fit(X_train_scaled, y_train)
    
    # Подбор порога по precision-recall на train (CV, без утечки)
    print(f"\nПодбор порога по кривой precision-recall (train, 5-fold CV):")
    threshold, threshold_info = tune_threshold(model, X_train_scaled, y_train, strategy='f1')
    save_threshold(threshold, threshold_info, os.path.join(model_dir, "threshold.json"))

    # Предсказания на тесте с подобранным порогом
    print(f"\nОценка на тестовой выборке (порог = {threshold:.4f}):")
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = predict_with_threshold(y_test_proba, threshold)

    test_metrics, test_cm = evaluate_model(
        y_test, y_test_pred, y_test_proba,
        f"{model_name} (Test)", threshold=threshold,
    )

    # Построение графиков
    plot_pr_curve(y_test, y_test_proba, f"{model_name}",
                  os.path.join(model_dir, "pr_curve.png"), threshold=threshold)
    plot_metrics_vs_threshold(y_test, y_test_proba, f"{model_name}",
                            os.path.join(model_dir, "metrics_vs_threshold.png"),
                            threshold=threshold)
    
    plot_confusion_matrix(test_cm, f"{model_name}",
                         os.path.join(model_dir, "confusion_matrix.png"))
    
    # Feature importance
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    plot_feature_importance(model, feature_names, model_name,
                           os.path.join(model_dir, "feature_importance.png"))
    
    # Сохранение feature importance в отдельный файл
    from utils import get_feature_importance
    importances = get_feature_importance(model, feature_names)
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    
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
    result = train_random_forest(X_train, X_test, y_train, y_test, base_dir)
    
    print("\nГотово!")

