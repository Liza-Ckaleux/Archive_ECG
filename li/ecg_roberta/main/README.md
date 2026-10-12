# ECG RoBERTa (HeartBERT)

Проект по обучению и применению языковой модели **RoBERTa** к сигналам ЭКГ: предобучение на большом корпусе (PTB-XL), дообучение на бинарную классификацию и веб-демо для загрузки EDF-файлов.

> **Важно:** ноутбуки рассчитаны на запуск в **Kaggle** (пути вида `/kaggle/input/...`). Для локального запуска замените пути к данным и артефактам моделей.

## Задача

- **Предобучение (HeartBERT):** токенизация дискретизированных фрагментов ЭКГ и обучение `RobertaForMaskedLM` (masked language modeling).
- **Дообучение:** бинарный классификатор **HeartBERT + BiLSTM** (`BertBiLSTMClassifier`) — классы **LFV** / **noLFV** (в веб-приложении: наличие / отсутствие **сниженной фракции выброса**).
- **Демо:** Django-приложение `ECGApp/` — загрузка EDF, визуализация 8 отведений, инференс обученной модели.

## Структура репозитория

| Файл / папка | Назначение |
|--------------|------------|
| `data-prep.ipynb` | Подготовка датасетов: `ecg_tune.pkl`, `ecg_data.pkl`, `ecg_data_combined.pkl` (PTB-XL + данные для fine-tune) |
| `heartbert.ipynb` | Обучение BPE-токенизатора (словарь ~52k) и предобучение RoBERTa (MLM) |
| `finetuning.ipynb` | Дообучение классификатора, метрики, сохранение весов (`state_dict.pt`, `model250.pt`) |
| `ECGApp/` | Веб-интерфейс (Django): загрузка EDF → предобработка → предсказание |

## Пайплайн

```
PTB-XL + клинические данные
        ↓
  data-prep.ipynb  →  ecg_data*.pkl, ecg_tune.pkl
        ↓
  heartbert.ipynb  →  токенизатор + HeartBERT (pt20k_250, …)
        ↓
  finetuning.ipynb →  BertBiLSTMClassifier + веса
        ↓
  ECGApp/          →  инференс по EDF (нужны model/ и state_dict.pt)
```

## Ноутбуки

### 1. `data-prep.ipynb`

- Обработка сигналов (neurokit2, ресемплинг, дискретизация).
- Формирование:
  - **`ecg_tune.pkl`** — ~1816 записей (сбалансированные классы для дообучения),
  - **`ecg_data.pkl`** — ~8979 записей,
  - **`ecg_data_combined.pkl`** — объединённый корпус (~19k) для предобучения.
- Источник для предобучения: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/).

### 2. `heartbert.ipynb`

1. **Токенизатор** — `SentencePieceBPETokenizer`, спецтокены `<s>`, `<pad>`, `</s>`, `<unk>`, `<cls>`, `<sep>`, `<mask>`.
2. **Модель** — `RobertaConfig` + `RobertaForMaskedLM`, MLM (`mlm_probability=0.15`).
3. Обучение на подготовленных pickle-файлах, сохранение чекпоинта (на Kaggle — в `/kaggle/working/`).

### 3. `finetuning.ipynb`

- Загрузка предобученного HeartBERT (`pt20k_250`).
- Архитектура: замороженный BERT + **BiLSTM** + линейный слой.
- Данные: `ecg_tune.pkl`, метки LFV (1) / noLFV (0).
- Обучение, PR-кривые, подбор порога, экспорт:
  - `state_dict.pt` — веса классификатора для `ECGApp`,
  - `model250.pt` / `model2501.pth` — полная модель.

## Зависимости (ноутбуки)

- Python 3.11+
- `torch`, `transformers`, `tokenizers`
- `numpy`, `pandas`, `scikit-learn`
- `neurokit2`, `matplotlib`

```bash
pip install torch transformers tokenizers numpy pandas scikit-learn neurokit2 matplotlib
```

## Веб-приложение ECGApp

Подробности — в [`ECGApp/README.md`](ECGApp/README.md).

**Для инференса нужны:**

- `ECGApp/ecg_app/model/` — веса HeartBERT и токенизатор,
- `ECGApp/ecg_app/state_dict.pt` — веса `BertBiLSTMClassifier` после `finetuning.ipynb`.

**Запуск:**

```bash
cd ECGApp
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Дополнительно для модели:
# pip install torch transformers
python manage.py migrate
python manage.py runserver
```

Откройте http://127.0.0.1:8000/ и загрузите **EDF** с отведениями: ECG I, II, V1–V6.

## Данные на Kaggle

В ноутбуках используются датасеты Kaggle

- PTB-XL,
- подготовленные pickle (`ecg_tune`, `finetuning-dataset`),
- предобученный HeartBERT (`pretrained_heartbert`).

## Репозиторий

https://gitlab.com/mmiracle.in/ecg_roberta


