import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    def __init__(self, I3D_feature_size = 400, hidden_rnn_layers = 3, hidden_rnn_nodes = 256, fc_dim = 128,
                 dropout_rate=0.3, output_size=11):
        super(LSTM_Model, self).__init__()

        self.input_size = I3D_feature_size
        self.hidden_rnn_layers = hidden_rnn_layers
        self.hidden_rnn_nodes = hidden_rnn_nodes
        self.fc_dim = fc_dim
        self.dropout_rate = dropout_rate
        self.output_size = output_size

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_rnn_nodes,
            num_layers=self.hidden_rnn_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.hidden_rnn_nodes, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.output_size)

    def forward(self, x):
        self.LSTM.flatten_parameters(x, None)
        rnn_out, h_n, h_c = self.LSTM(x, None)

        x = self.fc1(rnn_out[:,-1,:])
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training= self.train())
        x = self.fc2

        return x