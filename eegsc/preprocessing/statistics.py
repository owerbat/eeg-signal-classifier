from math import ceil
import numpy as np
from scipy.signal import periodogram, welch


class BaseStatisticsTransformer:
    def __init__(self, metrics: dict) -> None:
        self.sensors_num = 32
        self.metrics = metrics

        self.columns = []
        for name in self.metrics.keys():
            self.columns += [f'{name}_{i}' for i in range(self.sensors_num)]

    def transform(self, signals: np.ndarray):
        raise NotImplementedError()


class AmplitudeStatisticsTransformer(BaseStatisticsTransformer):
    def __init__(self) -> None:
        metrics = {
            'mean': lambda x: np.nanmean(x, axis=-1),
            'std': lambda x: np.nanstd(x, axis=-1),
            'median': lambda x: np.nanmedian(x, axis=-1),
            'min': lambda x: np.nanmin(x, axis=-1),
            'max': lambda x: np.nanmax(x, axis=-1),
        }

        super().__init__(metrics)

    def transform(self, signals: np.ndarray):
        return np.hstack([metric(signals) for _, metric in self.metrics.items()])


class FrequencyStatisticsTransformer(BaseStatisticsTransformer):
    def __init__(self,
                 psd_method: str = 'periodogram',
                 signal_duration: float = 10) -> None:
        if psd_method not in ('periodogram', 'welch'):
            raise ValueError('psd_method has to be one of the following: '
                             '`periodogram`, `welch`')

        self.psd_func = None
        if psd_method == 'periodogram':
            self.psd_func = periodogram
        elif psd_method == 'welch':
            self.psd_func = welch

        self.signal_duration = signal_duration

        metrics = {
            'psd_mean': lambda x: np.nanmean(x, axis=-1),
            'psd_std': lambda x: np.nanstd(x, axis=-1),
            'psd_median': lambda x: np.nanmedian(x, axis=-1),
            'psd_min': lambda x: np.nanmin(x, axis=-1),
            'psd_max': lambda x: np.nanmax(x, axis=-1),
        }

        super().__init__(metrics)

    def transform(self, signals: np.ndarray):
        psd = np.full((signals.shape[0], signals.shape[1], ceil(signals.shape[2] / 2)),
                      np.nan)

        for i, signal in enumerate(signals):
            actual_len = signal[0][~np.isnan(signal[0])].shape[0]
            actual_signal = signal[:, :actual_len]
            _, local_psd = self.psd_func(x=actual_signal,
                                         fs=actual_len / self.signal_duration,
                                         scaling='spectrum')
            psd[i, :, :local_psd.shape[1]] = local_psd

        return np.hstack([metric(psd) for _, metric in self.metrics.items()])
