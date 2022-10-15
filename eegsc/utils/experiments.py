import os
import numpy as np

from eegsc.preprocessing.filters import BandPassFilter
from eegsc.preprocessing.spectrum import SpectrumTransformer
from eegsc.utils.path import get_data_path, make_dir


def create_spectrum_dataset(data: dict,
                            bandpass_filter: BandPassFilter,
                            spectrum_transformer: SpectrumTransformer,
                            compute_stat: bool = True,
                            save: bool = True):
    hash = f'{bandpass_filter.order}_{spectrum_transformer.order}_' + \
           f'{spectrum_transformer.psd_method}_{compute_stat}'
    root_path = make_dir(os.path.join(get_data_path(), 'statistics_dataset', hash))
    statistics = []
    labels = []

    for i, (key, value) in enumerate(data.items()):
        statistic_path = os.path.join(root_path, f'{key}.npy')

        if os.path.exists(statistic_path):
            statistic = np.load(statistic_path)
        else:
            trials = value[2]
            filtered_trials = bandpass_filter.filter(trials)
            statistic = spectrum_transformer.transform(filtered_trials, compute_stat)

            if save:
                np.save(statistic_path, statistic)

        statistics.append(statistic)
        labels += [i] * statistic.shape[0]

    statistics = np.vstack(statistics)
    labels = np.array(labels, dtype=int)

    return statistics, labels


def create_sequence_dataset(data: dict,
                            bandpass_filter: BandPassFilter):
    seq_data = []
    labels = []

    for i, (_, value) in enumerate(data.items()):
        trials = value[2]
        filtered_trials = bandpass_filter.filter(trials)

        seq_data.append(filtered_trials)
        labels += [i] * filtered_trials.shape[0]

    seq_data = np.vstack(seq_data)
    labels = np.array(labels, dtype=int)

    return seq_data, labels
