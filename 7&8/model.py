import torch
from torch import nn


# Feedforward network that flattens MNIST (28 by 28) and classifies it to [0-9]
class FeedForwardMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

