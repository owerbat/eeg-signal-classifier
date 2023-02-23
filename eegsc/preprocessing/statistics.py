from math import ceil
import numpy as np
from scipy.signal import periodogram, welch

from eegsc.preprocessing.filters import FFTFilter


class BaseStatisticsTransformer:
    def __init__(self, metrics: dict) -> None:
        self.sensors_num = 32
        self.metrics = metrics
        self.support_channel = {
            'delta': True,
            'theta': True,
            'alpha': True,
            'alpha_wide': True,
            'beta': True,
            'gamma': True,
            'original': True,
        }

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

        self.support_channel['alpha_wide'] = False

    def transform(self, signals: np.ndarray):
        return np.hstack([metric(signals) for _, metric in self.metrics.items()])


class FrequencyStatisticsTransformer(BaseStatisticsTransformer):
    def __init__(self,
                 psd_method: str = 'periodogram',
                 signal_duration: float = 10,
                 metrics: dict = None) -> None:
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
        } if metrics is None else metrics

        super().__init__(metrics)

        self.support_channel['alpha_wide'] = False

    def _compute_psd(self, signals: np.ndarray):
        freqs = np.full(
            (signals.shape[0], signals.shape[1], ceil(signals.shape[2] / 2) + 1), np.nan)
        psd = np.full(
            (signals.shape[0], signals.shape[1], ceil(signals.shape[2] / 2) + 1), np.nan)

        for i, signal in enumerate(signals):
            actual_len = signal[0][~np.isnan(signal[0])].shape[0]
            actual_signal = signal[:, :actual_len]
            local_freqs, local_psd = self.psd_func(
                x=actual_signal,
                fs=actual_len / self.signal_duration,
                scaling='spectrum'
            )

            freqs[i, :, :local_psd.shape[1]] = local_freqs
            psd[i, :, :local_psd.shape[1]] = local_psd

        return freqs, psd

    def transform(self, signals: np.ndarray):
        _, psd = self._compute_psd(signals)
        return np.hstack([metric(psd) for _, metric in self.metrics.items()])


class IAFStatisticsTransformer(FrequencyStatisticsTransformer):
    def __init__(self,
                 psd_method: str = 'periodogram',
                 signal_duration: float = 10) -> None:
        def _center_of_mass(masses: np.ndarray, x: np.ndarray):
            result = np.zeros(masses.shape[:2])

            for i in range(masses.shape[0]):
                for j in range(masses.shape[1]):
                    result[i, j] = np.dot(np.nan_to_num(masses[i, j, :]),
                                          np.nan_to_num(x[i, j, :])) / \
                        np.nansum(masses[i, j, :])

            return result

        metrics = {
            'IAF': _center_of_mass,
        }

        super().__init__(psd_method, signal_duration, metrics)

        for channel in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'original']:
            self.support_channel[channel] = False
        self.support_channel['alpha_wide'] = True

    def transform(self, signals: np.ndarray):
        freqs, psd = self._compute_psd(signals)
        return np.hstack([metric(psd, freqs) for _, metric in self.metrics.items()])


class PAFStatisticsTransformer(FrequencyStatisticsTransformer):
    def __init__(self,
                 psd_method: str = 'periodogram',
                 signal_duration: float = 10) -> None:
        def _peak_freq(psd: np.ndarray, freqs: np.ndarray):
            result = np.zeros(psd.shape[:2])
            max_idxs = np.nanargmax(psd, axis=-1)

            for i in range(psd.shape[0]):
                for j in range(psd.shape[1]):
                    result[i, j] = freqs[i, j, max_idxs[i, j]]

            return result

        metrics = {
            'PAF': _peak_freq,
        }

        super().__init__(psd_method, signal_duration, metrics)

        for channel in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'original']:
            self.support_channel[channel] = False
        self.support_channel['alpha_wide'] = True

    def transform(self, signals: np.ndarray):
        freqs, psd = self._compute_psd(signals)
        return np.hstack([metric(psd, freqs) for _, metric in self.metrics.items()])


class MeanPSDStatisticsTransformer(FrequencyStatisticsTransformer):
    def __init__(self,
                 psd_method: str = 'periodogram',
                 signal_duration: float = 10) -> None:
        metrics = {
            'MeanPSD': lambda x: np.nanmean(x, axis=-1),
        }

        super().__init__(psd_method, signal_duration, metrics)

        for channel in ['delta', 'original', 'alpha_wide']:
            self.support_channel[channel] = False
        for channel in ['theta', 'alpha', 'beta', 'gamma']:
            self.support_channel[channel] = True

    def transform(self, signals: np.ndarray):
        _, psd = self._compute_psd(signals)
        return np.hstack([metric(psd) for _, metric in self.metrics.items()])


class TRPStatisticsTransformer(BaseStatisticsTransformer):
    def __init__(self,
                 min_norm_power: float = .2) -> None:
        def _task_related_power(signals: np.ndarray):
            result = np.zeros(signals.shape[:2])

            for i in range(signals.shape[0]):
                for j in range(signals.shape[1]):
                    squared = signals[i, j] ** 2
                    mid = len(squared) // 2
                    no_action = np.log(np.nanmean(squared[:mid]))
                    action = np.log(np.nanmean(squared[mid:]))
                    result[i, j] = action - no_action

            return result

        metrics = {
            'TRP': _task_related_power,
        }

        self.fft_filter = FFTFilter(min_norm_power)

        super().__init__(metrics)

        for channel in ['delta', 'original', 'alpha_wide']:
            self.support_channel[channel] = False
        for channel in ['theta', 'alpha', 'beta', 'gamma']:
            self.support_channel[channel] = True

    def transform(self, signals: np.ndarray):
        signals = self.fft_filter.filter(signals)
        return np.hstack([metric(signals) for _, metric in self.metrics.items()])
