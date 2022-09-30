import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
from typing import List, Union

from eegsc.utils.path import get_data_path


def _get_max_shape(raw_data: dict):
    max_shape = 0

    for person in raw_data['subs_ica']:
        for data_type in person.keys():
            times = person[data_type]['time']
            max_len = max([time.shape[0] for time in times])
            max_shape = max(max_shape, max_len)

    return max_shape


def _extract_data(raw_data: Union[dict, str], data_type: str, save: bool):
    info_path = os.path.join(get_data_path(), f'{data_type}_info.parquet')
    times_path = os.path.join(get_data_path(), f'{data_type}_times.npy')
    trials_path = os.path.join(get_data_path(), f'{data_type}_trials.npy')

    if os.path.exists(info_path) and os.path.exists(times_path) and \
        os.path.exists(trials_path):
        info = pd.read_parquet(info_path)
        times = np.load(times_path)
        trials = np.load(trials_path)

        return info, times, trials, raw_data

    raw_data = read_mat(raw_data) if isinstance(raw_data, str) else raw_data
    max_shape = _get_max_shape(raw_data)

    sensors_num = 32
    trials_num = sum([len(person[data_type]['time']) for person in raw_data['subs_ica']])

    times = np.full((trials_num, max_shape), np.nan)
    trials = np.full((trials_num, sensors_num, max_shape), np.nan)
    person_idxs = []
    trial_idxs = []

    idx = 0
    for person_idx, person in enumerate(raw_data['subs_ica']):
        person_times = person[data_type]['time']
        person_trials = person[data_type]['trial']

        for trial_idx, (time, trial) in enumerate(zip(person_times, person_trials)):
            length = len(time)

            times[idx, :length] = time
            trials[idx, :, :length] = trial
            person_idxs.append(person_idx)
            trial_idxs.append(trial_idx)

            idx += 1

    info = pd.DataFrame({
        'person_idx': np.array(person_idxs),
        'trial_idx': np.array(trial_idxs),
    })

    if save:
        info.to_parquet(info_path)
        np.save(times_path, times)
        np.save(trials_path, trials)

    return info, times, trials, raw_data


def read_raw(path: str,
             data_types: List[str] = ['left_real', 'right_real'],
             save: bool = True):
    raw_data = path
    result = {}

    for data_type in data_types:
        info, times, trials, raw_data = _extract_data(raw_data, data_type, save)
        result[data_type] = (info, times, trials)

    return result
