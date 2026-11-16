"""Classifier and raw-pixel interface for the MNIST attack experiment."""

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 61), nn.ReLU(), nn.Linear(61, 10),
        )

    def forward(self, normalized):
        return self.net(normalized)


class PixelsToLogits(nn.Module):
    def __init__(self, cnn, mean, std):
        super().__init__()
        self.cnn = cnn
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, pixels):
        return self.cnn((pixels - self.mean) / self.std)
