import torch.nn as nn

cnn = nn.Sequential(
    nn.Conv1d(400, 256, 3, stride = 2),
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 2),   # batch, chnel,w
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.Conv1d(256, 256, 3, stride = 2),
    nn.BatchNorm1d(256),
    nn.ReLU(),

    nn.AdaptiveAvgPool1d(3),

    nn.Flatten()
)


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = cnn
        self.rnn = nn.LSTM(768, 128, 2, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 48)


    def forward(self, x):
        batch_size, timesteps, C, W = x.size()
        c_in = x.view(batch_size * timesteps, C, W)

        # c_out: batch_size*timesteps, 256
        c_out = self.cnn(c_in)

        # try 1: lstm and fc
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.rnn(r_in)
        r_out = r_out.view(batch_size, timesteps, -1)
        out = self.fc1(r_out)
        out = self.fc2(out)

        # fc between segs


        return out
