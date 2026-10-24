"""Временное разбиение и единые группы оценки."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


VALID_START = pd.Timestamp('2022-04-22', tz='UTC')
TEST_START = pd.Timestamp('2022-05-01', tz='UTC')


def split_and_encode(events):
    time = pd.to_datetime(events.time_ms, unit='ms', utc=True)
    parts = {
        'train': events.loc[time < VALID_START].copy(),
        'validation': events.loc[(time >= VALID_START) & (time < TEST_START)].copy(),
        'test': events.loc[time >= TEST_START].copy(),
    }
    user_ids = np.sort(parts['train'].user_id.unique())
    item_ids = np.sort(parts['train'].video_id.unique())
    user_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_map = pd.Series(np.arange(len(item_ids)), index=item_ids)
    for frame in parts.values():
        frame['u'] = frame.user_id.map(user_map).fillna(-1).astype(int)
        frame['i'] = frame.video_id.map(item_map).fillna(-1).astype(int)
    return parts, user_ids, item_ids


def event_matrix(events, shape, values=None):
    """Сохраняет и явные нули: ALS учитывает все наблюдавшиеся показы."""
    if values is None:
        values = np.ones(len(events))
    return sparse.csr_matrix((values, (events.u, events.i)), shape=shape, dtype=float)


def positive_matrix(train, shape):
    return event_matrix(train.loc[train.is_relevant.eq(1)], shape)


@dataclass
class EvaluationSet:
    users: np.ndarray
    seen: sparse.csr_matrix
    relevant: sparse.csr_matrix


def make_cohort(heldout, history, train_positive, k=10):
    shape = train_positive.shape
    known_history = history.loc[history.u.ge(0) & history.i.ge(0)]
    seen = event_matrix(known_history, shape)
    seen.data[:] = 1
    relevant_events = heldout.loc[
        heldout.u.ge(0) & heldout.i.ge(0) & heldout.is_relevant.eq(1)
    ]
    relevant = event_matrix(relevant_events, shape)
    if seen.multiply(relevant).nnz:
        raise ValueError('Релевантные объекты уже присутствуют в истории')
    eligible = (
        (train_positive.getnnz(axis=1) > 0)
        & (relevant.getnnz(axis=1) > 0)
        & (shape[1] - seen.getnnz(axis=1) >= k)
    )
    return EvaluationSet(np.flatnonzero(eligible), seen, relevant)
