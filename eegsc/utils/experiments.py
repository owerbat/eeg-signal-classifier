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
        info, times, trials = value
        cut_times = np.full((times.shape[0], max_len), np.nan)
        cut_trials = np.full((trials.shape[0], trials.shape[1], max_len), np.nan)

        for i, (time, trial) in enumerate(zip(times, trials)):
            idxs = np.squeeze(np.argwhere(np.bitwise_and(~np.isnan(time),
                                                         time >= start_time)))
            cut_times[i, :len(idxs)] = time[idxs]
            cut_trials[i, :, :len(idxs)] = trial[:, idxs]

        data[key] = (info, cut_times, cut_trials)


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
    person_idxs = []

    _cut_by_time(data, start_time)

    for i, (key, value) in enumerate(data.items()):
        statistic_path = os.path.join(root_path, f'{key}.npy')
        info = value[0]

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
        person_idxs.append(info['person_idx'].to_numpy())

    statistics = np.vstack(statistics)
    labels = np.array(labels, dtype=int)
    person_idxs = np.hstack(person_idxs)

    return statistics, labels, person_idxs


def _cut_by_time_equal(data: dict, start_time: float):
    min_len = np.inf
    max_len, max_time = 0, 0

    for _, value in data.items():
        times = value[1]
        for time in times:
            idxs = np.argwhere(time != np.nan)

            min_len = min(min_len, len(idxs))
            max_len = max(max_len, len(idxs))

            max_time = max(max_time, time[idxs[-1]])

    if int((1 - start_time / max_time) * max_len) <= min_len:
        common_len = int((1 - start_time / max_time) * max_len)
    else:
        common_len = min_len

    for i, (key, value) in enumerate(data.items()):
        info, times, trials = value
        cut_times = np.zeros((times.shape[0], common_len))
        cut_trials = np.zeros((trials.shape[0], trials.shape[1], common_len))

        for i, (time, trial) in enumerate(zip(times, trials)):
            idxs = np.squeeze(np.argwhere(~np.isnan(time)))
            cut_times[i, :] = time[idxs][-common_len:]
            cut_trials[i, :, :] = trial[:, idxs][:, -common_len:]

        data[key] = (info, cut_times, cut_trials)


def create_sequence_dataset(data: dict,
                            bandpass_filter: BandPassFilter,
                            start_time: float = 0.0,
                            is_len_equal: bool = False,
                            return_time: bool = False):
    seq_data = []
    labels = []
    person_idxs = []
    times = []

    if is_len_equal:
        _cut_by_time_equal(data, start_time)
    else:
        _cut_by_time(data, start_time)

    for i, (_, value) in enumerate(data.items()):
        info, time, trials = value
        filtered_trials = bandpass_filter.filter(trials)

        seq_data.append(filtered_trials)
        labels += [i] * filtered_trials.shape[0]
        person_idxs.append(info['person_idx'].to_numpy())
        if return_time:
            times.append(time)

    seq_data = np.vstack(seq_data)
    labels = np.array(labels, dtype=int)
    person_idxs = np.hstack(person_idxs)
    if return_time:
        return seq_data, labels, person_idxs, np.vstack(times)
    else:
        return seq_data, labels, person_idxs
