import numpy as np
import torch
import torch.nn as nn


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
              criterion,
              optimizer,
              n_epochs: int = 10):
    model.to(model.device)
    model.train()

    running_loss = 0
    running_total = 0
    running_correct = 0

    for epoch in range(n_epochs):
        for trial, label in zip(x_train, y_train):
            actual_len = trial[0][~np.isnan(trial[0])].shape[0]
            trial = trial[:, :actual_len].T
            label = np.array([label])

            trial = torch.from_numpy(trial).to(model.device).float()
            label = torch.from_numpy(label).to(model.device)

            out = model(trial)
            # print(f'out: {out.size()}')
            # print(f'label: {label.size()}')
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            running_total += label.size(0)
            running_correct += (predicted == label).sum().item()

        print(f'Epoch {epoch + 1} | loss = {running_loss}, ' \
              f'accuracy = {running_correct / running_total}')

    return model