"""Собственный таргет: доля просмотра, лайк и подписка."""

import numpy as np
from scipy import sparse

from .ials import IALS


def engagement_target(events):
    completion = np.minimum(events.play_time_ms / events.duration_ms, 1.0)
    signal = 0.7 * completion + 0.2 * events.is_like + 0.1 * events.is_follow
    return ((1 - events.is_hate) * signal).to_numpy(dtype=float)


class EngagementALS(IALS):
    def fit(self, events, shape):
        targets = sparse.csr_matrix(
            (engagement_target(events), (events.u, events.i)), shape=shape,
        )
        return super().fit(targets)
