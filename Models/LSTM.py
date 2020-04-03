import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as u


class LSTM_Model(nn.Module):
    def __init__(self, I3D_feature_size, hidden_rnn_layers, hidden_rnn_nodes, fc_dim,
                 dropout_rate, output_size, bidirectional):
        super(LSTM_Model, self).__init__()

        self.input_size = I3D_feature_size
        self.hidden_rnn_layers = hidden_rnn_layers
        self.hidden_rnn_nodes = hidden_rnn_nodes
        self.fc_dim = fc_dim
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_rnn_nodes,
            num_layers=self.hidden_rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        fc_nodes = hidden_rnn_nodes * 2 if self.bidirectional else hidden_rnn_nodes
        self.fc1 = nn.Linear(fc_nodes, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.output_size)

    def forward(self, x):
        self.LSTM.flatten_parameters()
        rnn_out, (h_n, h_c) = self.LSTM(x, None)
        rnn_out, input_sizes = u.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        x = self.fc1(rnn_out[:,-1,:])
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training= self.training)
        x = self.fc2(x)

        return x