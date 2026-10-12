"""
Общие утилиты для обучения моделей
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)


def find_optimal_threshold(y_true, y_proba, strategy='f1', min_precision=None, min_recall=None):
    """
    Подбор порога по кривой precision-recall.

    strategy:
        'f1' — максимизация F1 (по умолчанию)
        'balanced' — максимизация (precision + recall) / 2
        'high_recall' — максимальный recall при precision >= min_precision
        'high_precision' — максимальная precision при recall >= min_recall
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    p = precision[1:]
    r = recall[1:]

    if len(thresholds) == 0:
        return 0.5, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'strategy': strategy}

    if strategy == 'f1':
        scores = 2 * p * r / (p + r + 1e-12)
    elif strategy == 'balanced':
        scores = (p + r) / 2
    elif strategy == 'high_recall':
        min_p = 0.5 if min_precision is None else min_precision
        mask = p >= min_p
        scores = np.where(mask, r, -1.0)
    elif strategy == 'high_precision':
        min_r = 0.5 if min_recall is None else min_recall
        mask = r >= min_r
        scores = np.where(mask, p, -1.0)
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    best_idx = int(np.argmax(scores))
    threshold = float(thresholds[best_idx])
    info = {
        'precision': float(p[best_idx]),
        'recall': float(r[best_idx]),
        'f1': float(2 * p[best_idx] * r[best_idx] / (p[best_idx] + r[best_idx] + 1e-12)),
        'strategy': strategy,
    }
    return threshold, info


def predict_with_threshold(y_proba, threshold):
    """Бинарные предсказания по вероятности положительного класса."""
    return (y_proba >= threshold).astype(int)


def get_cv_probabilities(model, X, y, n_splits=5, random_state=42):
    """Out-of-fold вероятности на train для подбора порога без утечки."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in skf.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        proba[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
    return proba


def tune_threshold(model, X_train, y_train, strategy='f1', n_splits=5,
                   min_precision=None, min_recall=None):
    """Подбор порога на train через кросс-валидацию."""
    y_proba_cv = get_cv_probabilities(model, X_train, y_train, n_splits=n_splits)
    threshold, info = find_optimal_threshold(
        y_train, y_proba_cv, strategy=strategy,
        min_precision=min_precision, min_recall=min_recall,
    )
    info['threshold'] = threshold
    return threshold, info


def save_threshold(threshold, info, save_path):
    """Сохранение подобранного порога и метрик на train."""
    payload = {'threshold': threshold, **info}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Порог сохранён в {save_path}: {threshold:.4f} "
          f"(P={info['precision']:.4f}, R={info['recall']:.4f}, F1={info['f1']:.4f})")


def evaluate_model(y_true, y_pred, y_proba=None, model_name="", threshold=None):
    """Оценка модели с различными метриками для обоих классов и макро-усредненные"""
    # Общие метрики
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
    }
    
    # Метрики для каждого класса и макро-усредненные
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Метрики для класса 0
    metrics['Precision_Class_0'] = precision[0]
    metrics['Recall_Class_0'] = recall[0]
    metrics['F1_Score_Class_0'] = f1[0]
    metrics['Support_Class_0'] = int(support[0])
    
    # Метрики для класса 1
    if len(precision) > 1:
        metrics['Precision_Class_1'] = precision[1]
        metrics['Recall_Class_1'] = recall[1]
        metrics['F1_Score_Class_1'] = f1[1]
        metrics['Support_Class_1'] = int(support[1])
    else:
        metrics['Precision_Class_1'] = 0.0
        metrics['Recall_Class_1'] = 0.0
        metrics['F1_Score_Class_1'] = 0.0
        metrics['Support_Class_1'] = 0
    
    # Макро-усредненные метрики
    metrics['Precision_Macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Recall_Macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['F1_Score_Macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if threshold is not None:
        metrics['Threshold'] = threshold

    # PR-AUC (площадь под кривой precision-recall)
    if y_proba is not None:
        try:
            metrics['PR-AUC'] = average_precision_score(y_true, y_proba)
        except Exception:
            metrics['PR-AUC'] = 0.0
    
    print(f"\n{'='*60}")
    print(f"Метрики для {model_name}:")
    print(f"{'='*60}")
    print(f"{'Accuracy':25s}: {metrics['Accuracy']:.4f}")
    print(f"\nМетрики по классам:")
    print(f"  Класс 0:")
    print(f"    Precision: {metrics['Precision_Class_0']:.4f}")
    print(f"    Recall:    {metrics['Recall_Class_0']:.4f}")
    print(f"    F1-Score:  {metrics['F1_Score_Class_0']:.4f}")
    print(f"    Support:   {metrics['Support_Class_0']}")
    print(f"  Класс 1:")
    print(f"    Precision: {metrics['Precision_Class_1']:.4f}")
    print(f"    Recall:    {metrics['Recall_Class_1']:.4f}")
    print(f"    F1-Score:  {metrics['F1_Score_Class_1']:.4f}")
    print(f"    Support:   {metrics['Support_Class_1']}")
    print(f"\nМакро-усредненные метрики:")
    print(f"  Precision: {metrics['Precision_Macro']:.4f}")
    print(f"  Recall:    {metrics['Recall_Macro']:.4f}")
    print(f"  F1-Score:  {metrics['F1_Score_Macro']:.4f}")
    if threshold is not None:
        print(f"  Порог:     {metrics['Threshold']:.4f}")
    if 'PR-AUC' in metrics:
        print(f"  PR-AUC:    {metrics['PR-AUC']:.4f}")
    
    print(f"\nМатрица ошибок:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Дополнительная диагностика
    print(f"\nДиагностика предсказаний:")
    print(f"  Уникальные предсказания: {np.unique(y_pred)}")
    print(f"  Распределение предсказаний: {np.bincount(y_pred) if len(np.unique(y_pred)) <= 2 else 'много классов'}")
    print(f"  Уникальные истинные метки: {np.unique(y_true)}")
    print(f"  Распределение истинных меток: {np.bincount(y_true) if len(np.unique(y_true)) <= 2 else 'много классов'}")
    
    if y_proba is not None:
        print(f"  Диапазон вероятностей: [{np.min(y_proba):.4f}, {np.max(y_proba):.4f}]")
        print(f"  Средняя вероятность класса 1: {np.mean(y_proba):.4f}")
    
    print(f"\nОтчет о классификации:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return metrics, cm


def plot_pr_curve(y_true, y_proba, model_name, save_path=None, threshold=None):
    """Построение кривой precision-recall с отметкой выбранного порога."""
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        baseline = np.mean(y_true)

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        plt.axhline(y=baseline, color='k', linestyle='--',
                    label=f'Baseline (prevalence = {baseline:.3f})')

        if threshold is not None and len(thresholds) > 0:
            idx = int(np.argmin(np.abs(thresholds - threshold)))
            plt.scatter(
                recall[idx + 1], precision[idx + 1],
                color='red', s=80, zorder=5,
                label=f'Threshold = {threshold:.3f}',
            )

        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(fontsize=14)
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR кривая сохранена в {save_path}")

        plt.close()
    except Exception as e:
        print(f"Не удалось построить PR кривую: {e}")


def plot_metrics_vs_threshold(y_true, y_proba, model_name, save_path=None, threshold=None):
    """Precision, recall и F1 в зависимости от порога классификации."""
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        p = precision[1:]
        r = recall[1:]
        f1 = 2 * p * r / (p + r + 1e-12)

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, p, label='Precision', linewidth=2)
        plt.plot(thresholds, r, label='Recall', linewidth=2)
        plt.plot(thresholds, f1, label='F1', linewidth=2, linestyle='--')

        if threshold is not None and len(thresholds) > 0:
            idx = int(np.argmin(np.abs(thresholds - threshold)))
            plt.axvline(x=threshold, color='red', linestyle=':', linewidth=2,
                        label=f'Optimal threshold = {threshold:.3f}')
            plt.scatter(
                [threshold, threshold, threshold],
                [p[idx], r[idx], f1[idx]],
                color='red', s=60, zorder=5,
            )

        plt.xlabel('Threshold', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        plt.tick_params(labelsize=14)
        plt.title(f'Precision / Recall / F1 vs Threshold - {model_name}')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График метрик vs порог сохранён в {save_path}")

        plt.close()
    except Exception as e:
        print(f"Не удалось построить график метрик vs порог: {e}")


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Визуализация матрицы ошибок"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     annot_kws={'size': 22},
                     xticklabels=['0', '1'],
                     yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=24)
    plt.ylabel('True Label', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Матрица ошибок сохранена в {save_path}")
    
    plt.close()


def get_feature_importance(model, feature_names):
    """Извлечение важности признаков из модели"""
    if hasattr(model, 'feature_importances_'):
        # Для Random Forest и других tree-based моделей
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Для Logistic Regression, SVM и других линейных моделей
        # Берем абсолютные значения коэффициентов
        coef = model.coef_
        if coef.ndim > 1:
            # Для бинарной классификации берем первый (или единственный) класс
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)
    else:
        # Если нет способа извлечь важность, возвращаем нули
        importances = np.zeros(len(feature_names))
    
    return importances


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Построение гистограммы важности признаков"""
    importances = get_feature_importance(model, feature_names)
    
    # Сортируем по важности
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Создаем график
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(sorted_features)), sorted_importances, color='steelblue')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Самые важные сверху
    
    # Добавляем значения на столбцы
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Гистограмма важности признаков сохранена в {save_path}")
    
    plt.close()

