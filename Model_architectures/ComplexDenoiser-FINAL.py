import torch
import torch.nn as nn

class ComplexDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=9, padding=4)
        )
    def forward(self, x):
        return self.net(x)