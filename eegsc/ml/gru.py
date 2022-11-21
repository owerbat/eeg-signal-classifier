from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import train_net, predict_net


class GRUNet(nn.Module):
    def __init__(self, input_size: int,
                       hidden_size: int,
                       n_layers: int,
                       n_classes: int,
                       fc_size: int = 0,
                       uniform_hidden_init: bool = False,
                       device: str = 'cpu') -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.uniform_hidden_init = uniform_hidden_init
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        if fc_size < 1:
            self.fc = nn.Linear(hidden_size, n_classes)
        else:
            self.fc = nn.Sequential(nn.Linear(hidden_size, fc_size),
                                    nn.ReLU(),
                                    nn.Linear(fc_size, n_classes))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
        if self.uniform_hidden_init:
            h0.uniform_(-np.sqrt(self.hidden_size), np.sqrt(self.hidden_size))

        out, _ = self.gru(x, h0)
        out = self.fc(out[-1:, :])
        out = self.softmax(out)

        return out


class StackingGRUNet(nn.Module):
    def __init__(self, input_size: int,
                       hidden_size: int,
                       fc_size: int,
                       n_layers: int,
                       n_classes: int,
                       final_fc_size: int = 0,
                       uniform_hidden_init: bool = False,
                       compute_avg: bool = False,
                       device: str = 'cpu') -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.uniform_hidden_init = uniform_hidden_init
        self.compute_avg = compute_avg
        self.device = device

        self.sensors_num = 32
        assert input_size % self.sensors_num == 0, 'input_size is not correct'
        self.spectrum_size = int(input_size / self.sensors_num)

        fc_input_size = 2 * hidden_size if compute_avg else hidden_size

        self.grus = [nn.GRU(self.sensors_num, hidden_size, n_layers).to(device)
                     for _ in range(self.spectrum_size)]
        self.fcs = [nn.Sequential(nn.Linear(fc_input_size, fc_size).to(device),
                                  nn.ReLU())
                    for _ in range(self.spectrum_size)]
        if final_fc_size < 1:
            self.final_fc = nn.Linear(fc_size * self.spectrum_size, n_classes).to(device)
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(fc_size * self.spectrum_size, final_fc_size).to(device),
                nn.ReLU(),
                nn.Linear(final_fc_size, n_classes).to(device)
            )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        outs = []
        for i in range(self.spectrum_size):
            h0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
            if self.uniform_hidden_init:
                h0.uniform_(-np.sqrt(self.hidden_size), np.sqrt(self.hidden_size))

            start_idx = i * self.sensors_num
            end_idx = (i + 1) * self.sensors_num

            out, _ = self.grus[i](x[:, start_idx: end_idx], h0)
            if self.compute_avg:
                out = torch.cat([out[-1, :], torch.mean(out, dim=0)])
                out = torch.unsqueeze(out, dim=0)
            else:
                out = out[-1:, :]
            out = self.fcs[i](out)

            outs.append(out)

        common_out = torch.cat(outs, dim=1)
        common_out = self.final_fc(common_out)
        out = self.softmax(common_out)

        return out


def _gru_trial_proc_func(trial: np.ndarray):
    actual_len = trial[0][~np.isnan(trial[0])].shape[0]
    return trial[:, :actual_len].T


def train_gru(model: Any,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              criterion: Any,
              optimizer: Any,
              n_epochs: int = 10):
    return train_net(trial_proc_func=_gru_trial_proc_func,
                     model=model,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     criterion=criterion,
                     optimizer=optimizer,
                     n_epochs=n_epochs)


def predict_gru(model: GRUNet,
                x_test: np.ndarray,
                y_test: np.ndarray = None,
                criterion: Any = None,
                compute_loss: bool = False):
    return predict_net(trial_proc_func=_gru_trial_proc_func,
                       model=model,
                       x_test=x_test,
                       y_test=y_test,
                       criterion=criterion,
                       compute_loss=compute_loss)
