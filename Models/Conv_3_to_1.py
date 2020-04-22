import torch.nn as nn
import math

encoder = nn.Sequential(
    nn.Conv1d(400, 256, 3, stride = 3),
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 3),   # batch, chnel,w
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 3),
    nn.BatchNorm1d(256),
    nn.ReLU(),

   nn.AdaptiveAvgPool1d(3),

    nn.Flatten()
)

single_logits_classifier = nn.Sequential(
    nn.Linear(3 * 256, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 48)
)

three_logits_classifier = nn.Sequential(
    nn.Linear(3 * 256 * 3, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 48)
)






