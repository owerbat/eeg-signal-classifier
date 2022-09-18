import numpy as np

from eegsc.preprocessing.filters import BandPassFilter
from eegsc.preprocessing.statistics import AmplitudeStatisticsTransformer


class SpectrumTransformer:
    def __init__(self,
                 order: int = 2,
                 signal_duration: float = 10) -> None:
        self.filters = {
            'delta': BandPassFilter(order, 1, 4, signal_duration),
            'theta': BandPassFilter(order, 4, 8, signal_duration),
            'alpha': BandPassFilter(order, 8, 12, signal_duration),
            'beta': BandPassFilter(order, 12, 20, signal_duration),
            'gamma': BandPassFilter(order, 20, 50, signal_duration),
        }
        self.ampl_transformer = AmplitudeStatisticsTransformer()

        self.columns = []
        for name in ['original'] + list(self.filters.keys()):
            self.columns += [f'{name}_{col}' for col in self.ampl_transformer.columns]

    def transform(self, signals: np.ndarray):
        spectrum_signals = {'original': signals}
        for name, filter in self.filters.items():
            spectrum_signals[name] = filter.filter(signals)

        spectrum_ampl_statistics = [
            self.ampl_transformer.transform(spectrum_signal)
            for _, spectrum_signal in spectrum_signals.items()
        ]

        spectrum_freq_statistics = []

        return np.hstack(spectrum_ampl_statistics + spectrum_freq_statistics)
