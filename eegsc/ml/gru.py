import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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


def train_gru(model: GRUNet,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              criterion,
              optimizer,
              n_epochs: int = 10):
    model.to(model.device)

    for epoch in range(n_epochs):
        train_loss = 0
        train_total = 0
        train_correct = 0

        model.train()

        for trial, label in tqdm(zip(x_train, y_train)):
            actual_len = trial[0][~np.isnan(trial[0])].shape[0]
            trial = trial[:, :actual_len].T
            label = np.array([label])

            trial = torch.from_numpy(trial).to(model.device).float()
            label = torch.from_numpy(label).to(model.device).long()

            out = model(trial)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

        _, test_loss, test_accuracy = predict_gru(model,
                                                  x_test,
                                                  y_test,
                                                  criterion,
                                                  compute_loss=True)

        print(f'Epoch {epoch + 1} | train_loss = {train_loss}, ' \
              f'train_accuracy = {train_correct / train_total} | ' \
              f'test_loss = {test_loss}, test_accuracy = {test_accuracy}')

    return model


def predict_gru(model: GRUNet,
                x_test: np.ndarray,
                y_test: np.ndarray = None,
                criterion=None,
                compute_loss: bool = False):
    model.to(model.device)
    model.eval()

    result = []
    test_loss = 0
    test_total = 0
    test_correct = 0

    with torch.no_grad():
        for trial, label in tqdm(zip(x_test, y_test)):
            actual_len = trial[0][~np.isnan(trial[0])].shape[0]
            trial = trial[:, :actual_len].T
            label = np.array([label])

            trial = torch.from_numpy(trial).to(model.device).float()
            label = torch.from_numpy(label).to(model.device).long()

            out = model(trial)
            _, predicted = torch.max(out.data, 1)

            test_loss += criterion(out, label).item()
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

            result.append(predicted.item())

    if compute_loss:
        return np.array(result), test_loss, test_correct / test_total
    else:
        return np.array(result)
