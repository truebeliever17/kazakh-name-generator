import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout):
        super(LSTMGenerator, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(vocab_size, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        out, prev_state = self.lstm(x, prev_state)

        out = out.squeeze(0)

        out = self.dropout(out)

        return self.fc(out), prev_state

    def init_state(self, device):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))
