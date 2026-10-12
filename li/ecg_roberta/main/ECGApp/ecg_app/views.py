import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.contrib import messages
import pyedflib
from scipy.signal import find_peaks, butter, filtfilt
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import io
import base64
import tempfile
from safetensors.torch import load_file
from .models import BertBiLSTMClassifier
from .preprocessing import preprocess_ecg_signal

# Настройка логирования
logger = logging.getLogger(__name__)

class BertBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes=1, bert_frozen_layers=3, input_size=768, lstm_hidden_size=128):
        super(BertBiLSTMClassifier, self).__init__()

        # Путь к предобученной модели в локальной директории
        bert_model_path = os.path.join(settings.BASE_DIR, 'ecg_app', 'model')
        logger.info(f'Загрузка HeartBERT из {bert_model_path}')
        
        if not os.path.exists(bert_model_path):
            raise FileNotFoundError(f'Предобученная модель HeartBERT не найдена в {bert_model_path}')
            
        # HeartBERT model with frozen layers
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=bert_model_path)

        modules = [self.bert.embeddings, *self.bert.encoder.layer[:bert_frozen_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
                
        # Bi-LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        # BERT forward pass
        bert_output = self.bert(input_ids)[0]
        # Bi-LSTM forward pass
        lstm_out, _ = self.lstm(bert_output)
        # Use the last hidden state from Bi-LSTM
        lstm_last_hidden_state = lstm_out[:, -1, :]
        # Classification using fully connected layer
        res = self.fc(lstm_last_hidden_state)
        # logit = self.fc(lstm_last_hidden_state)
        # res = self.sigmoid(logit)
        return res.squeeze(-1)

def read_edf_record(file):
    """Чтение EDF файла из памяти без сохранения на диск"""
    try:
        # Сохраняем временно в памяти
        logger.info('Начинаем чтение файла')
        file_content = file.read()
        logger.info(f'Размер файла: {len(file_content)} байт')
        
        # Проверяем первые байты файла
        if len(file_content) >= 8:
            header = file_content[:8].decode('ascii', errors='ignore')
            logger.info(f'Заголовок файла: {header}')
            if not header.startswith('0       '):
                logger.error('Неверный формат EDF файла: неправильный заголовок')
                return None, None, None
        
        # Создаем временный файл
        logger.info('Создаем временный файл')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name
            logger.info(f'Временный файл создан: {temp_file_path}')
        
        try:
            logger.info('Открываем файл через pyedflib')
            f = pyedflib.EdfReader(temp_file_path)
            logger.info('Файл успешно открыт')
            
            try:
                logger.info('Получаем информацию о каналах')
                n_channels = f.signals_in_file
                logger.info(f'Количество каналов: {n_channels}')
                channel_labels = f.getSignalLabels()
                logger.info(f'Метки каналов: {channel_labels}')
                
                logger.info('Читаем сигналы')
                signals = []
                for i in range(n_channels):
                    try:
                        logger.info(f'Начинаем чтение канала {i}: {channel_labels[i]}')
                        signal = f.readSignal(i)
                        logger.info(f'Канал {i} успешно прочитан, размер: {len(signal)}')
                        signals.append(signal)
                    except Exception as e:
                        logger.error(f'Ошибка при чтении канала {i}: {str(e)}')
                        logger.error(f'Тип ошибки: {type(e)}')
                        import traceback
                        logger.error(f'Traceback: {traceback.format_exc()}')
                        return None, None, None
                
                signals = np.vstack(signals)
                logger.info(f'Размерность сигналов: {signals.shape}')
                
                sfreq = f.getSampleFrequency(0)
                logger.info(f'Частота дискретизации: {sfreq}')
                
                f.close()
                logger.info('Файл успешно закрыт')
                return signals, channel_labels, sfreq
            except Exception as e:
                logger.error(f'Ошибка при чтении данных: {str(e)}')
                logger.error(f'Тип ошибки: {type(e)}')
                import traceback
                logger.error(f'Traceback: {traceback.format_exc()}')
                f.close()
                return None, None, None
        finally:
            # Удаляем временный файл
            try:
                logger.info('Удаляем временный файл')
                os.unlink(temp_file_path)
                logger.info('Временный файл успешно удален')
            except Exception as e:
                logger.error(f'Ошибка при удалении временного файла: {str(e)}')
    except Exception as e:
        logger.error(f'Ошибка при чтении EDF файла: {str(e)}')
        logger.error(f'Тип ошибки: {type(e)}')
        import traceback
        logger.error(f'Traceback: {traceback.format_exc()}')
        return None, None, None

def extract_periods(signal, sfreq, num_periods=1, distance_sec=0.6):
    """Извлечение периодов кардиосигнала"""
    peaks, _ = find_peaks(signal, distance=int(distance_sec * sfreq * 0.8))
    if len(peaks) < num_periods + 1:
        raise ValueError("Недостаточно пиков")
    periods = []
    for i in range(num_periods):
        start = peaks[i]
        end = peaks[i+1]
        half = int((end-start)/2)
        periods.append(signal[end-half:end+half])
    return periods

def process_ecg_data(signals, channel_labels, sfreq, target_channels):
    """Обработка ECG данных"""
    lowcut = 0.6
    highcut = 40
    order = 1
    nyquist_freq = 0.5 * sfreq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    
    if signals is None or channel_labels is None:
        raise ValueError("Сигналы не загружены")
    
    if not all(ch in channel_labels for ch in target_channels):
        missing = [ch for ch in target_channels if ch not in channel_labels]
        raise ValueError(f"Отсутствуют каналы: {missing}")
    
    ecg_pieces = []
    for ch in target_channels:
        idx = channel_labels.index(ch)
        ch_data = signals[idx]
        b, a = butter(order, [low, high], btype='bandpass')
        ch_dat = filtfilt(b, a, ch_data)
        
        try:
            periods = extract_periods(ch_dat, sfreq, num_periods=1)
            ecg_pieces.append(periods[0])
        except Exception as e:
            logger.error(f"Ошибка при извлечении периодов: {str(e)}")
            raise ValueError(f"Не удалось извлечь периоды для канала {ch}")
    
    return np.hstack(ecg_pieces).tolist()

def create_ecg_plot(signals, channel_labels, times):
    """Создание графика ECG"""
    plt.figure(figsize=(15, 10))
    for i, lead in enumerate(channel_labels):
        plt.subplot(4, 2, i+1)
        plt.plot(times, signals[i])
        plt.title(f'Отведение {lead}')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
    
    plt.tight_layout()
    
    # Сохраняем график в base64 вместо файла
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def home(request):
    if request.method == 'POST':
        try:
            if 'ecg_file' not in request.FILES:
                messages.error(request, 'Файл не был загружен')
                return render(request, 'ecg_app/home.html')
            
            ecg_file = request.FILES['ecg_file']
            if not ecg_file.name.endswith('.edf'):
                messages.error(request, 'Пожалуйста, загрузите файл в формате EDF')
                return render(request, 'ecg_app/home.html')
            
            # Чтение и обработка файла
            signals, channel_labels, sfreq = read_edf_record(ecg_file)
            if signals is None:
                messages.error(request, 'Ошибка при чтении файла EDF')
                return render(request, 'ecg_app/home.html')
            
            # Проверка необходимых отведений
            target_channels = ['ECG I', 'ECG II', 'ECG V1', 'ECG V2', 
                             'ECG V3', 'ECG V4', 'ECG V5', 'ECG V6']
            
            missing_leads = [lead for lead in target_channels if lead not in channel_labels]
            if missing_leads:
                messages.error(request, f'Отсутствуют следующие отведения: {", ".join(missing_leads)}')
                return render(request, 'ecg_app/home.html')
            
            # Создание графика
            times = np.arange(signals.shape[1]) / sfreq
            plot_base64 = create_ecg_plot(signals, target_channels, times)
            
            # Обработка сигнала для модели
            try:
                processed_signal = process_ecg_data(signals, channel_labels, sfreq, target_channels)
                # Предобработка сигнала
                preprocessed_text = preprocess_ecg_signal(processed_signal)
            except ValueError as e:
                messages.error(request, f'Ошибка обработки сигнала: {str(e)}')
                return render(request, 'ecg_app/home.html')
            
            # Загрузка и использование модели
            try:
                # Путь к HeartBERT модели
                bert_model_path = os.path.join(settings.BASE_DIR, 'ecg_app', 'model')
                logger.info(f'Загрузка HeartBERT из {bert_model_path}')
                
                if not os.path.exists(bert_model_path):
                    raise FileNotFoundError(f'Файл HeartBERT не найден: {bert_model_path}')
                
                # Загружаем предобученную модель BERT
                bert_model = AutoModel.from_pretrained(bert_model_path)
                bert_model.eval()
                
                # Путь к model360
                model360_path = os.path.join(settings.BASE_DIR, 'ecg_app', 'state_dict.pt')
                logger.info(f'Загрузка model360 из {model360_path}')
                
                if not os.path.exists(model360_path):
                    raise FileNotFoundError(f'Файл state_dict.pt не найден: {model360_path}')
                
                # Загружаем model360
                model360_state_dict = torch.load(model360_path, map_location=torch.device('cpu'))
                
                # Создаем новую модель и загружаем веса
                model360 = BertBiLSTMClassifier()
                model360.load_state_dict(model360_state_dict)
                model360.eval()
                
                # Токенизируем входные данные
                tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
                inputs = tokenizer(
                    [preprocessed_text],  # Передаем предобработанный текст
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                logger.info(f'Размерность токенизированных данных: {inputs["input_ids"].shape}')
                
                with torch.no_grad():
                    # Пропускаем через model360
                    output = model360(inputs["input_ids"])
                    logger.info(f'Выход model360: {output}')
                
                predicted = (torch.sigmoid(output) > 0.5).long()
                is_sick = predicted.item() == 1
                diagnosis = "Обнаружена сниженная фракция выброса" if is_sick else "Сниженная фракция выброса отсутствует"
                logger.info(f'Диагноз: {diagnosis}')
                
            except FileNotFoundError as e:
                logger.error(f'Ошибка при загрузке модели: {str(e)}')
                diagnosis = "Ошибка анализа: модель не найдена"
                is_sick = False
            except Exception as e:
                logger.error(f'Ошибка при работе модели: {str(e)}')
                logger.error(f'Тип ошибки: {type(e)}')
                import traceback
                logger.error(f'Traceback: {traceback.format_exc()}')
                diagnosis = "Ошибка анализа"
                is_sick = False
            
            context = {
                'plot_url': f"data:image/png;base64,{plot_base64}",
                'diagnosis': diagnosis,
                'is_sick': is_sick
            }
            return render(request, 'ecg_app/result.html', context)
            
        except Exception as e:
            logger.error(f'Общая ошибка: {str(e)}')
            messages.error(request, f'Произошла ошибка: {str(e)}')
            return render(request, 'ecg_app/home.html')
    
    return render(request, 'ecg_app/home.html')