import numpy as np
import scipy


class BandPassFilter:
    def __init__(self,
                 order: int = 2,
                 min_freq: float = 1,
                 max_freq: float = 50,
                 signal_duration: float = 10) -> None:
        self.order = order
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.signal_duration = signal_duration

    def filter(self, signals: np.ndarray):
        result = np.full(signals.shape, np.nan)

        for i, signal in enumerate(signals):
            actual_len = signal[0][~np.isnan(signal[0])].shape[0]
            sos = scipy.signal.butter(N=self.order,
                                      Wn=(self.min_freq, self.max_freq),
                                      btype='bandpass',
                                      output='sos',
                                      fs=actual_len / self.signal_duration)
            filtered = scipy.signal.sosfilt(sos, signal)
            result[i, :, :] = filtered

        return result


class FFTFilter:
    def __init__(self, min_norm_power: float = .2) -> None:
        self.min_norm_power = min_norm_power

    def _filter_1d(self, signal: np.ndarray):
        fft = scipy.fft.rfft(signal)
        norm_fft = np.abs(fft) / np.abs(fft).max()

        filtered_fft = fft.copy()
        filtered_fft[np.where(norm_fft < self.min_norm_power)[0]] = 0
        filtered_signal = scipy.fft.irfft(filtered_fft, n=len(signal))

        return filtered_signal

    def filter(self, signals: np.ndarray):
        result = np.zeros(signals.shape)

        for i, sample in enumerate(signals):
            for j, signal in enumerate(sample):
                result[i, j, :] = self._filter_1d(signal)

        return result
