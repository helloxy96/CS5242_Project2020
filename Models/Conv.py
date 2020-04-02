import torch.nn as nn


model = nn.Sequential(
    nn.Conv1d(400, 256, 3, stride = 2),
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 2),   # batch, chnel,w
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 2),
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.AdaptiveAvgPool1d(20),

    nn.Flatten(),   
    nn.Linear(5120, 1024), 
    nn.BatchNorm1d(1024), 
    nn.ReLU(),

    nn.Linear(1024, 256), 
    nn.BatchNorm1d(256), 
    nn.ReLU(),

    nn.Dropout(0.2),
    nn.Linear(256, 48)
)