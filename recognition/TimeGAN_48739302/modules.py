from typing import LiteralString

import torch.nn as nn


class Embedder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers=2, activation_fun=nn.Sigmoid()
    ):
        super().__init__()

        self.rnn_cell = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, x):
        h, _ = self.rnn_cell(x)
        h = self.fc(h)
        return self.act_fun(h)


class Recoverer(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()

        self.rnn_cell = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn_cell(x)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(
        self, noise_dim, hidden_dim, num_layers=2, activation_fun=nn.Sigmoid()
    ):
        super().__init__()

        self.rnn_cell = nn.GRU(
            noise_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, x):
        h, _ = self.rnn_cell(x)
        h = self.fc(h)
        h = self.act_fun(h)
        return h


class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, activation_fun=nn.Sigmoid()):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, h):
        s, _ = self.rnn(h)
        s = self.fc(s)
        s = self.act_fun(s)
        return s


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        d, _ = self.rnn(h)
        d = self.fc(d)
        return d
