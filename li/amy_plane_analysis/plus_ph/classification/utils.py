"""
Общие утилиты для обучения моделей
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_fscore_support
)


def evaluate_model(y_true, y_pred, y_proba=None, model_name=""):
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
    
    # ROC-AUC
    if y_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['ROC-AUC'] = 0.0
    
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
    if 'ROC-AUC' in metrics:
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
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


def plot_roc_curve(y_true, y_proba, model_name, save_path=None):
    """Построение ROC кривой"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(fontsize=16)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC кривая сохранена в {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"Не удалось построить ROC кривую: {e}")


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

