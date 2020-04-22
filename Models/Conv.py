import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 3, chopnum = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, stride = 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.chopnum = chopnum

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2, self.relu2
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        chopnum = self.chopnum
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, chopnum//2:-math.ceil(chopnum/2)].contiguous() # x fit into a right size

        return out

class Encoder(nn.Module):
    def __init__(self, n_inputs = 100):
        super(Encoder, self).__init__()

        chopnum1 = (100 - ((100 - 3) // 3 + 1))
        chopnum2 = (33 - ((33 - 3) // 3 + 1))
        chopnum3 = (11 - ((11 - 3) // 3 + 1))
        # chopnum4 = (5 - ((5 - 3) // 3 + 1))

        layers = [
            ResidualBlock(400, 256, 3, stride = 3, chopnum=chopnum1),
            ResidualBlock(256, 256, 3, stride = 3, chopnum=chopnum2),
            ResidualBlock(256, 256, 3, stride = 3, chopnum=chopnum3),
            
            nn.AdaptiveAvgPool1d(3),

            nn.Flatten(),

            nn.Linear(768, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(1024, 48)
            # ResidualBlock(256, 256, 3, stride = 3, chopnum=chopnum4),
        ]

        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        

    def forward(self, x):
        out = self.network(x)

        return self.flatten(out)