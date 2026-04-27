import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_samples=9000):
        super().__init__()
        self._features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            _flat = self._features(torch.zeros(1, 2, input_samples)).shape[1]
        self._head = nn.Sequential(
            nn.Linear(_flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._head(self._features(x))
