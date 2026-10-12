# ECG Anomaly Detection using Autoencoders and XGBoost

Проект для обнаружения аномалий в ЭКГ сигналах с использованием автоэнкодеров и ансамбля XGBoost.

## Описание

Двухэтапный подход:

1. **Обучение автоэнкодера** для каждого отведения (i, ii, v1–v6) только на нормальных образцах
2. **Извлечение признаков** из автоэнкодера (ошибка реконструкции MSE + латентный вектор)
3. **Обучение XGBoost** на объединённых признаках всех отведений
4. **Визуализация** ошибок реконструкции и multilead raw ECG для интерпретации

## Структура проекта

```
Autoencoder/
├── models/
│   └── autoencoder.py              # Архитектура автоэнкодера
├── scripts/
│   ├── prepare_train_zero.py       # train/test pickle по отведениям
│   ├── prepare_train2.py           # train2 pickle для ансамбля
│   ├── prepare_ptbxl_records500.py   # дополнение train нормальными PTB-XL
│   └── regenerate_feature_importance_ru.py
├── configs/
│   └── training_params.yaml
├── train_autoencoder.py            # Обучение AE для одного отведения
├── assembly_xgboost.py             # Ансамбль XGBoost по всем отведениям
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `torch` — автоэнкодер
- `xgboost` — финальный классификатор
- `scikit-learn` — метрики
- `pandas`, `numpy`, `scipy`, `pyedflib` — данные и предобработка

## Быстрый запуск

### 1. Подготовка данных

Pickle-файлы с колонками `signal`, `label`, `patient_id`:

- `train{lead}.pkl` — обучение AE (норма)
- `test{lead}.pkl` — валидация/тест AE
- `train2{lead}.pkl` — обучение XGBoost (оба класса)

Строка `i` во всех файлах отведений соответствует одному пациенту.

### 2. Обучение автоэнкодера (для каждого отведения)

```bash
python train_autoencoder.py \
    --train datasets12/trainv1.pkl \
    --test datasets12/testv1.pkl \
    --out artifacts_v1
```

Параметры по умолчанию читаются из `configs/training_params.yaml`.

Результаты в `artifacts_{lead}/`:
- `ae_model.pt`
- `ae_train_loss.png`
- `roc_curves.png`
- `reconstruction_error_histogram.png`
- `ae_report.txt`

### 3. Обучение ансамбля XGBoost

```bash
python assembly_xgboost.py
```

Скрипт:
- загружает AE из `artifacts_{lead}`
- извлекает признаки MSE + latent по всем 8 отведениям
- обучает XGBoost на 136 признаках
- сохраняет метрики и графики в `artifacts_ensemble_xgb*` 

Для патологий в формате Ph (`X_train.pkl`, `X_test.pkl`) задайте в YAML:

```yaml
dataset_name: "ph"   # ph | lef | amy
dataset_dir: "/path/to/Ph_datasets"
```

Результаты сохраняются в `artifacts_ensemble_xgb_{dataset_name}`.

## Конфигурация

Пример `configs/training_params.yaml`:

```yaml
epochs: 60
batch_size: 32
latent_dim: 16
lr: 0.0005
val_fraction: 0.3333333333333333
seed: 42

n_estimators: 300
learning_rate: 0.01
max_depth: 8

generate_raw_ecg_figures: true
n_periods: 5
```

## Архитектура

### Автоэнкодер
- **Энкодер:** Linear → BatchNorm → LeakyReLU → Dropout (1024 → 512 → 256 → 16)
- **Декодер:** Linear → ReLU (16 → 256 → 512 → 1024 → input_dim)
- **Обучение:** только норма, Smooth L1 Loss, Adam (lr = 0.0005)

### XGBoost
- Вход: 8 × (1 MSE + 16 latent) = **136 признаков**
- `objective: binary:logistic`
- `eval_metric: logloss, auc`

## Метрики

- Accuracy, Precision, Recall, F1
- ROC AUC
- Confusion Matrix
- Feature importance по отведениям

## Выходные файлы ансамбля

- `ensemble_xgboost_model.joblib`
- `ensemble_roc_curve.png`
- `ensemble_confusion_matrix.png`
- `ensemble_feature_importance.png`
- `training_curves.png`
- `raw_ecg_healthy_*.png`, `raw_ecg_sick_*.png`
- `ensemble_report.txt`
