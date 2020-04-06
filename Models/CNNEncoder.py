import torch.nn as nn
import torch


class CnnEncoder(nn.Module):
    def __init__(self):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=2, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=2, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU())

        self.pool = nn.AdaptiveMaxPool1d(5)

    def forward(self, x_frames):
        cnn_embed = []
        batchsize = x_frames.size(0)
        for i in range(x_frames.size(1)):  # batch, frames, data_feature
            x = x_frames[:, i, :].view(batchsize, 1, -1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(batchsize, -1)
            cnn_embed.append(x)
        cnn_embed = torch.stack(cnn_embed, dim=0).transpose_(0, 1)
        return cnn_embed