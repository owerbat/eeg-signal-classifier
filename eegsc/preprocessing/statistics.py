import numpy as np


class AmplitudeStatisticsTransformer:
    def __init__(self) -> None:
        self.sensors_num = 32
        self.metrics = {
            'mean': lambda x: np.nanmean(x, axis=-1),
            'std': lambda x: np.nanstd(x, axis=-1),
            'median': lambda x: np.nanmedian(x, axis=-1),
            'min': lambda x: np.nanmin(x, axis=-1),
            'max': lambda x: np.nanmax(x, axis=-1),
        }

        self.columns = []
        for name in self.metrics.keys():
            self.columns += [f'{name}_{i}' for i in range(self.sensors_num)]

    def transform(self, signals: np.ndarray):
        return np.hstack([metric(signals) for _, metric in self.metrics.items()])
