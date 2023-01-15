from typing import Any, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm


def train_test_split(data: np.ndarray,
                     labels: np.ndarray,
                     person_idxs: np.ndarray,
                     test_size: float = .2,
                     test_persons: list = [],
                     random_state: int = None):
    rng = np.random.default_rng(random_state)

    if len(test_persons):
        test_person_idxs = test_persons
    else:
        unique_persons = np.unique(person_idxs)
        n_persons = int(len(unique_persons) * test_size)
        test_person_idxs = rng.choice(unique_persons, n_persons, replace=False)

    train_idxs = np.where(~np.isin(person_idxs, test_person_idxs))[0]
    test_idxs = np.where(np.isin(person_idxs, test_person_idxs))[0]

    x_train = data[train_idxs]
    y_train = labels[train_idxs]
    x_test = data[test_idxs]
    y_test = labels[test_idxs]

    rng.shuffle(x_train)
    rng.shuffle(y_train)
    rng.shuffle(x_test)
    rng.shuffle(y_test)

    return x_train, x_test, y_train, y_test


def cross_val_score(data: np.ndarray,
                    labels: np.ndarray,
                    person_idxs: np.ndarray,
                    model: Any,
                    metric: Callable,
                    n_test_persons: int = 3,
                    random_state: int = None):
    rng = np.random.default_rng(random_state)

    unique_persons = np.unique(person_idxs)
    rng.shuffle(unique_persons)

    n_persons = len(unique_persons)
    n_iters = n_persons // n_test_persons + bool(n_persons % n_test_persons)

    persons = []
    train_metrics = []
    test_metrics = []

    for i in tqdm(range(n_iters)):
        start_idx = i * n_test_persons
        end_idx = min((i + 1) * n_test_persons, n_persons)

        test_persons = unique_persons[start_idx: end_idx]
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            labels,
                                                            person_idxs,
                                                            test_persons=test_persons,
                                                            random_state=random_state)

        model.fit(x_train, y_train)

        persons.append(test_persons)
        train_metrics.append(metric(y_train, model.predict(x_train)))
        test_metrics.append(metric(y_test, model.predict(x_test)))

    return pd.DataFrame({'test_persons': persons,
                         'train_metrics': train_metrics,
                         'test_metrics': test_metrics})
