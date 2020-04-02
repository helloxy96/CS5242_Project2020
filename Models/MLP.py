import torch.nn as nn


model = nn.Sequential(
    nn.Linear(400, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 48)
)