import numpy as np

from eegsc.preprocessing.filters import BandPassFilter
from eegsc.preprocessing.statistics import AmplitudeStatisticsTransformer, \
    FrequencyStatisticsTransformer


class SpectrumTransformer:
    def __init__(self,
                 order: int = 2,
                 signal_duration: float = 10,
                 psd_method: str = 'periodogram') -> None:
        self.order = order
        self.psd_method = psd_method

        self.filters = {
            'delta': BandPassFilter(order, 1, 4, signal_duration),
            'theta': BandPassFilter(order, 4, 8, signal_duration),
            'alpha': BandPassFilter(order, 8, 12, signal_duration),
            'beta': BandPassFilter(order, 12, 20, signal_duration),
            'gamma': BandPassFilter(order, 20, 50, signal_duration),
        }
        self.ampl_transformer = AmplitudeStatisticsTransformer()
        self.freq_transformer = FrequencyStatisticsTransformer(
            psd_method=psd_method,
            signal_duration=signal_duration
        )

        self.transformers = [self.ampl_transformer, self.freq_transformer]

        self.columns = []
        for transformer in self.transformers:
            for name in ['original'] + list(self.filters.keys()):
                self.columns += [f'{name}_{col}' for col in transformer.columns]

    def transform(self, signals: np.ndarray, compute_stat: bool = True):
        spectrum_signals = {'original': signals}
        for name, filter in self.filters.items():
            spectrum_signals[name] = filter.filter(signals)

        if compute_stat:
            spectrum_statistics = []

            for transformer in self.transformers:
                spectrum_statistics += [
                    transformer.transform(spectrum_signal)
                    for _, spectrum_signal in spectrum_signals.items()
                ]

            return np.hstack(spectrum_statistics)
        else:
            return np.concatenate(
                [spectrum_signal for _, spectrum_signal in spectrum_signals.items()],
                axis=1
            )
