# src/model.py
import torch.nn as nn
import torch

class GestureLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
