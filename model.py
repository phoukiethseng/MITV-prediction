import torch
from torch import nn
from einops.layers.torch import Rearrange

class MITVLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MITVLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.base = nn.Sequential(
            Rearrange('b l f -> b f l'),
            # nn.BatchNorm1d(hidden_size * 2),
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=1, kernel_size=1),
            nn.ReLU(),
            # nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=1),
            # nn.ReLU(),
            Rearrange('b f l -> b l f'),
        )
        self.multiplier = nn.Sequential(
            Rearrange('b l f -> b f l'),
            # nn.BatchNorm1d(hidden_size * 2),
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=1, kernel_size=1),
            nn.ReLU(),
            # nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=1),
            # nn.ReLU(),
            Rearrange('b f l -> b l f'),
        )
        self.adder = nn.Sequential(
            Rearrange('b l f -> b f l'),
            # nn.BatchNorm1d(hidden_size * 2),
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=1, kernel_size=1),
            nn.Tanh(),
            # nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=1),
            # nn.ReLU(),
            Rearrange('b f l -> b l f'),
        )


    def forward(self, X, seq_len):
        batch_size = len(X)
        padded_seq = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
        out, _ = self.lstm(padded_seq)
        b = self.base(out)
        m = self.adder(out)
        a = self.multiplier(out)

        out = (b*m) + a
        out = nn.utils.rnn.unpad_sequence(out, torch.hstack(seq_len), batch_first=True)
        out = torch.nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=0)
        return out

