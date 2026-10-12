from django.db import models
import torch
import torch.nn as nn
from transformers import AutoModel

class ECGRecord(models.Model):
    file = models.FileField(upload_to='ecg_files/')
    upload_date = models.DateTimeField(auto_now_add=True)
    diagnosis = models.CharField(max_length=100, blank=True)
    is_sick = models.BooleanField(default=False)
    
    def __str__(self):
        return f"ECG Record {self.id} - {self.upload_date}"

class BertBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes=1, bert_frozen_layers=3, input_size=768, lstm_hidden_size=128):
        super(BertBiLSTMClassifier, self).__init__()
        
        # HeartBERT model with frozen layers
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

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
        return res.squeeze(-1) 