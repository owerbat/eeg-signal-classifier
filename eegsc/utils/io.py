import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
from typing import List

from eegsc.utils.path import get_data_path


def _get_max_shape(raw_data: dict):
    max_shape = 0

    for person in raw_data['subs_ica']:
        for data_type in person.keys():
            times = person[data_type]['time']
            max_len = max([time.shape[0] for time in times])
            max_shape = max(max_shape, max_len)

    return max_shape


def _extract_data(raw_data: dict, data_type: str, max_shape: int, save: bool):
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
        info.to_parquet(os.path.join(get_data_path(), f'{data_type}_info.parquet'))
        np.save(os.path.join(get_data_path(), f'{data_type}_times.npy'), times)
        np.save(os.path.join(get_data_path(), f'{data_type}_trials.npy'), trials)

    return info, times, trials


def read_raw(path: str,
             data_types: List[str] = ['right_real', 'left_real'],
             save: bool = False):
    raw_data = read_mat(path)
    max_shape = _get_max_shape(raw_data)

    result = {}
    for data_type in data_types:
        result[data_type] = _extract_data(raw_data, data_type, max_shape, save)
    return result
