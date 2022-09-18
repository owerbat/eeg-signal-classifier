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
