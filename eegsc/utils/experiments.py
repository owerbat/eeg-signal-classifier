import os
import numpy as np

from eegsc.preprocessing.filters import BandPassFilter
from eegsc.preprocessing.spectrum import SpectrumTransformer
from eegsc.utils.path import get_data_path, make_dir


def _cut_by_time(data: dict, start_time: float):
    if start_time <= 0:
        return

    max_len = 0

    for _, value in data.items():
        times = value[1]
        for time in times:
            idxs = np.argwhere(np.bitwise_and(time != np.nan, time >= start_time))
            max_len = max(max_len, len(idxs))

    if max_len == 0:
        raise ValueError(f'start_time is bigger than max time')

    for i, (key, value) in enumerate(data.items()):
        times, trials = value[1], value[2]
        cut_times = np.full((times.shape[0], max_len), np.nan)
        cut_trials = np.full((trials.shape[0], trials.shape[1], max_len), np.nan)

        for i, (time, trial) in enumerate(zip(times, trials)):
            idxs = np.squeeze(np.argwhere(np.bitwise_and(time != np.nan,
                                                         time >= start_time)))
            cut_times[i, :len(idxs)] = time[idxs]
            cut_trials[i, :, :len(idxs)] = trial[:, idxs]

        data[key] = (value[0], cut_times, cut_trials)


def create_spectrum_dataset(data: dict,
                            bandpass_filter: BandPassFilter,
                            spectrum_transformer: SpectrumTransformer,
                            compute_stat: bool = True,
                            start_time: float = 0.0,
                            save: bool = True):
    hash = f'{bandpass_filter.order}_{spectrum_transformer.order}_' + \
           f'{spectrum_transformer.psd_method}_{compute_stat}_{start_time}'
    root_path = make_dir(os.path.join(get_data_path(), 'statistics_dataset', hash))
    statistics = []
    labels = []

    _cut_by_time(data, start_time)

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
                            bandpass_filter: BandPassFilter,
                            start_time: float = 0.0):
    seq_data = []
    labels = []

    _cut_by_time(data, start_time)

    for i, (_, value) in enumerate(data.items()):
        trials = value[2]
        filtered_trials = bandpass_filter.filter(trials)

        seq_data.append(filtered_trials)
        labels += [i] * filtered_trials.shape[0]

    seq_data = np.vstack(seq_data)
    labels = np.array(labels, dtype=int)

    return seq_data, labels
