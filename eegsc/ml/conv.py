from math import floor
from typing import Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import train_net, predict_net


def _compute_out_shape(height: int,
                       width: int,
                       kernel_size: tuple,
                       stride: tuple,
                       padding: tuple):
    out_height = floor((height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
    out_width = floor((width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
    return out_height, out_width


class BaseUnit(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 kernel_size: Union[int, tuple] = 5,
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = None,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.in_height = height
        self.in_width = width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else self.kernel_size // 2
        self.device = device

        int_to_tuple = lambda x: x if isinstance(x, tuple) else (x, x)
        self.out_height, self.out_width = _compute_out_shape(
            self.in_height,
            self.in_width,
            int_to_tuple(self.kernel_size),
            int_to_tuple(self.stride),
            int_to_tuple(self.padding)
        )

    def forward(self, input):
        raise NotImplementedError()


class ConvUnit(BaseUnit):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 height: int,
                 width: int,
                 kernel_size: Union[int, tuple] = 5,
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = None,
                 device: str = 'cpu') -> None:
        super().__init__(height=height,
                         width=width,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              device=self.device)

    def forward(self, input):
        return F.relu(self.conv(input))


class PoolUnit(BaseUnit):
    def __init__(self,
                 height: int,
                 width: int,
                 kernel_size: Union[int, tuple] = 5,
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = None,
                 device: str = 'cpu') -> None:
        super().__init__(height=height,
                         width=width,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         device=device)

        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 padding=self.padding)

    def forward(self, input):
        return self.pool(input)


class ConvNet(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 n_classes: int,
                 kernel_size: int = 5,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.device = device

        self.sensors_num = 32
        assert input_shape[0] % self.sensors_num == 0, 'input_shape is not correct'
        self.spectrum_size = input_shape[0] // self.sensors_num

        in_channels = self.spectrum_size
        out_channels = max(self.spectrum_size // 2, 1)
        self.conv1 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.sensors_num,
                              width=self.input_shape[1],
                              kernel_size=self.kernel_size,
                              stride=(1, 3),
                              padding=self.padding,
                              device=self.device)

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.conv2 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.conv1.out_height,
                              width=self.conv1.out_width,
                              kernel_size=self.kernel_size,
                              stride=(1, 3),
                              padding=self.padding,
                              device=self.device)

        self.pool1 = PoolUnit(height=self.conv2.out_height,
                              width=self.conv2.out_width,
                              kernel_size=3,
                              stride=(1, 2),
                              padding=1,
                              device=self.device)

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.conv3 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.pool1.out_height,
                              width=self.pool1.out_width,
                              kernel_size=self.kernel_size,
                              stride=(1, 3),
                              padding=self.padding,
                              device=self.device)

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.conv4 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.conv3.out_height,
                              width=self.conv3.out_width,
                              kernel_size=self.kernel_size,
                              stride=(1, 3),
                              padding=self.padding,
                              device=self.device)

        self.pool2 = PoolUnit(height=self.conv4.out_height,
                              width=self.conv4.out_width,
                              kernel_size=3,
                              stride=(2, 2),
                              padding=1,
                              device=self.device)

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.conv5 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.pool2.out_height,
                              width=self.pool2.out_width,
                              kernel_size=self.kernel_size,
                              stride=(2, 2),
                              padding=self.padding,
                              device=self.device)

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.conv6 = ConvUnit(in_channels=in_channels,
                              out_channels=out_channels,
                              height=self.conv5.out_height,
                              width=self.conv5.out_width,
                              kernel_size=self.kernel_size,
                              stride=(2, 2),
                              padding=self.padding,
                              device=self.device)

        self.fc_size = out_channels * self.conv6.out_height * self.conv6.out_width
        self.fc = nn.Linear(self.fc_size, self.n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.pool1(self.conv2(self.conv1(x)))
        out = self.pool2(self.conv4(self.conv3(out)))
        out = self.conv6(self.conv5(out))

        out = self.fc(torch.flatten(out, 1))
        out = self.softmax(out)

        return out


def _conv_trial_proc_func(trial: np.ndarray):
    sensors_num = 32
    assert trial.shape[0] % sensors_num == 0, 'trial.shape is not correct'
    spectrum_size = trial.shape[0] // sensors_num

    proc_trial = np.zeros((1, spectrum_size, sensors_num, trial.shape[1]))
    for i in range(spectrum_size):
        proc_trial[0, i] = trial[i * sensors_num: (i + 1) * sensors_num, :]

    return np.nan_to_num(proc_trial, nan=0)


def train_gru(model: Any,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              criterion: Any,
              optimizer: Any,
              n_epochs: int = 10):
    return train_net(trial_proc_func=_conv_trial_proc_func,
                     model=model,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     criterion=criterion,
                     optimizer=optimizer,
                     n_epochs=n_epochs)


def predict_gru(model: Any,
                x_test: np.ndarray,
                y_test: np.ndarray = None,
                criterion: Any = None,
                compute_loss: bool = False):
    return predict_net(trial_proc_func=_conv_trial_proc_func,
                       model=model,
                       x_test=x_test,
                       y_test=y_test,
                       criterion=criterion,
                       compute_loss=compute_loss)
