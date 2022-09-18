import numpy as np


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

    def filter(self, signal: np.ndarray):
        result = np.full(signal.shape, np.nan)

        for i, s in enumerate(signal):
            dirty_signal = s[~np.isnan(s)]
            sos = signal.butter(N=self.order,
                                Wn=(self.min_freq, self.max_freq),
                                btype='bandpass',
                                output='sos',
                                fs=dirty_signal.shape[0] / self.signal_duration)
            filtered = signal.sosfilt(sos, dirty_signal)
            result[i, :filtered.shape[0]] = filtered

        return result
