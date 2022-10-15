import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class GRUNet(nn.Module):
    def __init__(self, input_size: int,
                       hidden_size: int,
                       n_layers: int,
                       n_classes: int,
                       device: str = 'cpu') -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[-1:, :])
        out = self.softmax(out)

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
